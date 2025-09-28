# -*- coding:utf-8 -*-
import os
import argparse
import torch
import sys
import numpy as np
from tensorboardX import SummaryWriter

# ==== 项目内依赖 ====
from networks.common.ema import EMA

# 追加工程根目录到 sys.path
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from networks.common.seed import seed_all
from networks.common.config import CFG
from networks.common.dataset import get_dataset
from networks.common.model import get_model
from networks.common.io_tools import dict_to, _create_directory   # ★ 新增 _create_directory
from networks.common.metrics import Metrics
import networks.common.checkpoint as checkpoint
from loguru import logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser(description='SSA-SC validate only (DDP)')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='/home/ubuntu/Code/kzw/base_double/SSC_configs/examples/SSA_SC_MREF.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--dset_root',
        dest='dataset_root',
        default='/home/ubuntu/Dataset/kzw/SSC/sscbench-kitti/data',
        metavar='DATASET',
        help='path to dataset root folder',
        type=str,
    )
    # 保存预测结果的输出根目录（与参考脚本一致的目录组织）
    parser.add_argument(
        '--out_path',
        dest='output_path',
        default=None,
        metavar='OUT_PATH',
        help='path to folder where predictions will be saved (if set, predictions will be written)',
        type=str,
    )
    # 从环境变量读取 local_rank（torchrun 会设置）
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    return parser.parse_args()


def _save_batch_predictions(scores_dict, indices, val_loader, out_path_root, inv_remap_lut, log_each=False):
    """
    参考你给的保存代码，将一个 batch 的预测结果落盘。
    - scores_dict: model(data, stat=...) 返回的第 0 项（字典），内部包含 'pred_semantic_1_1' 等
    - indices: 与 dataloader 返回的 indices 一一对应
    - val_loader: 用来取 filepaths 等元信息
    - out_path_root: 输出根目录（与参考代码同结构）
    - inv_remap_lut: 语义标签反映射 LUT（uint16）
    """
    if 'pred_semantic_1_1' not in scores_dict:
        # 如果键名不同，可以在这里扩展；默认优先保存 1_1 语义尺度
        return

    # 取语义预测，做 argmax -> [B, D, H, W]（与参考代码同）
    pred_sem = torch.argmax(scores_dict['pred_semantic_1_1'], dim=1).data.cpu().numpy()
    curr_index = 0

    for score in pred_sem:
        # [D,H,W] -> [D,W,H] 再 reshape 到一维，保持与标注写法一致
        score = np.moveaxis(score, [0, 1, 2], [0, 2, 1]).reshape(-1).astype(np.uint16)
        score = inv_remap_lut[score].astype(np.uint16)

        # 用 dataloader 暴露的 filepaths + indices 组装目标文件名
        input_filename = val_loader.dataset.filepaths['3D_OCCUPANCY'][indices[curr_index]]
        filename, _ = os.path.splitext(os.path.basename(input_filename))
        sequence = os.path.dirname(input_filename).split('/')[-2]
        out_filename = os.path.join(out_path_root, 'sequences', sequence, 'predictions', filename + '.label')

        # 创建目录并写出
        _create_directory(os.path.dirname(out_filename))
        score.tofile(out_filename)

        if log_each:
            logger.info('=> Sequence {} - File {} saved'.format(sequence, os.path.basename(out_filename)))

        curr_index += 1


@torch.no_grad()
def run_validation(model, val_loader, _cfg, logger, tbwriter, metrics, ema, out_path_root=None):
    """
    单次完整验证流程（无 epoch 循环）。
    复用 Metrics 与模型的 compute_loss / get_target / get_scales。
    当 out_path_root 非空时，将预测结果按参考代码的格式保存为 .label 文件。
    """
    ema.apply_shadow()  # 应用 EMA 权重（若无 EMA 训练，shadow 等于当前权重）

    device = torch.device('cuda')
    torch.cuda.empty_cache()
    model.module.eval()

    # 重置度量
    metrics.reset_evaluator()
    metrics.losses_track.restart_validation_losses()

    # 反映射 LUT（参考保存脚本）
    # 注意：示例代码里用的是 dset.dataset.get_inv_remap_lut()
    # 在本验证脚本里，val_loader.dataset 即为实际数据集封装，暴露相同接口
    inv_remap_lut = val_loader.dataset.get_inv_remap_lut() if hasattr(val_loader.dataset, 'get_inv_remap_lut') \
        else val_loader.dataset.dataset.get_inv_remap_lut()

    for t, (data, indices) in enumerate(val_loader):
        data = dict_to(data, device)
        # 与原验证一致：使用 'val' 前向
        scores = model(data, stat='val')  # 可能返回 (dict, other...)，下面按原逻辑取 scores[0]
        loss = model.module.compute_loss(scores, data)

        # 累计损失与指标（注意原逻辑里 metrics.add_batch 用的是 scores[0]）
        metrics.losses_track.update_validaiton_losses(loss)
        metrics.add_batch(prediction=scores[0], target=model.module.get_target(data))

        # ★ 保存预测（所有 rank 均会保存自己这份，避免重复；日志只在 rank 0 打印）
        if out_path_root is not None:
            _save_batch_predictions(
                scores_dict=scores[0],
                indices=indices,
                val_loader=val_loader,
                out_path_root=out_path_root,
                inv_remap_lut=inv_remap_lut,
                log_each=(dist.get_rank() == 0)
            )

        # 日志
        if (t + 1) % _cfg._dict['VAL']['SUMMARY_PERIOD'] == 0 and dist.get_rank() == 0:
            loss_str = ', '.join([f'{k}={v:.6f}' for k, v in loss.items()])
            logger.info(f'=> Iteration [{t + 1}/{len(val_loader)}], Val Losses: {loss_str}')

    # 汇总
    val_count = max(1, metrics.losses_track.validation_iteration_counts)
    total_loss = metrics.losses_track.validation_losses['total'] / val_count

    if dist.get_rank() == 0:
        logger.info(f'=> [Validation - Total Loss = {total_loss}]')

        # 写入 TensorBoard
        for l_key in metrics.losses_track.validation_losses:
            tbwriter.add_scalar(
                f'validation_loss_epoch/{l_key}',
                metrics.losses_track.validation_losses[l_key].item() / val_count,
                0
            )

        for scale in metrics.evaluator.keys():
            miou = metrics.get_semantics_mIoU(scale).item()
            iou_occ = metrics.get_occupancy_IoU(scale).item()
            tbwriter.add_scalar(f'validation_performance/{scale}/mIoU', miou, 0)
            tbwriter.add_scalar(f'validation_performance/{scale}/IoU', iou_occ, 0)
            logger.info(f'=> [Scale {scale}: mIoU = {miou:.6f} - IoU = {iou_occ:.6f}]')

        logger.info('=> Validation set class-wise IoU (scale=1_1):')
        for i in range(1, metrics.nbr_classes):
            class_name = val_loader.dataset.dataset_config['labels'][val_loader.dataset.dataset_config['learning_map_inv'][i]]
            class_score = metrics.evaluator['1_1'].getIoU()[1][i]
            logger.info('    => {}: {:.6f}'.format(class_name, class_score))

    ema.restore()  # 可选：恢复为原始权重

    return {
        'loss': float(total_loss),
        'mIoU_1_1': float(metrics.get_semantics_mIoU('1_1')),
        'IoU_1_1': float(metrics.get_occupancy_IoU('1_1')),
    }


def main():
    # 与训练脚本保持一致
    torch.backends.cudnn.enabled = False
    seed_all(7777)

    args = parse_args()

    # 初始化分布式
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # 读取配置
    _cfg = CFG()
    _cfg.from_config_yaml(args.config_file)

    # 替换数据路径
    if args.dataset_root is not None:
        _cfg._dict['DATASET']['ROOT_DIR'] = args.dataset_root

    # TensorBoard
    tb_path = os.path.join(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'metrics_val_only')
    tbwriter = SummaryWriter(logdir=tb_path)

    # Logger
    if dist.get_rank() == 0:
        logger.add(os.path.join(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'validate_only_{time}.log'), encoding='utf-8')
        logger.info('============ Validate-only routine ============')
        logger.info(f'=> Config: {args.config_file}')
        logger.info(f'=> Dataset root: {_cfg._dict["DATASET"]["ROOT_DIR"]}')
        if args.output_path is not None:
            logger.info(f'=> Predictions will be saved to: {args.output_path}')

    # 数据与模型
    dataset = get_dataset(_cfg)
    if dist.get_rank() == 0:
        logger.info('=> Building model...')
    model = get_model(_cfg, dataset['train'].dataset)  # 用 train.dataset 以便获得相同 heads/scales

    # DDP & SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ===== 关键：用同一行接口直接加载权重（优化器、调度器传 None）=====
    if dist.get_rank() == 0:
        logger.info('=> Loading checkpoint via checkpoint.load(...)')
    model, _, _, epoch = checkpoint.load_double(
        model,
        None,  # optimizer 传 None
        None,  # scheduler 传 None
        _cfg._dict['STATUS']['RESUME'],
        _cfg._dict['STATUS']['LAST'],
        logger
    )
    if dist.get_rank() == 0:
        logger.info(f'=> Checkpoint loaded (epoch = {epoch})')

    # 构建 Metrics 与 EMA，运行验证
    val_loader = dataset['val']
    metrics = Metrics(val_loader.dataset.nbr_classes, len(val_loader), model.module.get_scales())
    metrics.reset_evaluator()
    metrics.losses_track.set_validation_losses(model.module.get_validation_loss_keys())

    ema = EMA(model, 0.999)
    ema.register()  # 影子权重 = 当前已加载权重

    results = run_validation(
        model,
        val_loader,
        _cfg,
        logger,
        tbwriter,
        metrics,
        ema,
        out_path_root=args.output_path  # ★ 传入保存路径（可为 None）
    )

    # 等待所有 rank 写完文件再收尾
    dist.barrier()

    if dist.get_rank() == 0:
        logger.info('=> ============ Validation Complete ============')
        logger.info('=> [mIoU (1_1) = {m:.6f} | IoU (1_1) = {i:.6f} | Loss = {l:.6f}]'
                    .format(m=results['mIoU_1_1'], i=results['IoU_1_1'], l=results['loss']))
        logger.info('=> Writing config snapshot to output folder')
    _cfg.finish_config()

    if dist.get_rank() == 0:
        logger.info('=> Validate-only routine completed.')


if __name__ == '__main__':
    main()
