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


# ★新增：参数/内存统计工具
def count_params_and_buffers(model: torch.nn.Module):
    """统计可训练参数数量、buffers 数量及其内存占用（字节）。"""
    # 只统计 requires_grad 的参数作为“参数量”
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    # 也统计全部参数（有些项目也会报告 total params）
    n_all_params = sum(p.numel() for p in model.parameters())
    all_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # buffers（如BN均值/方差等）
    n_buffers = sum(b.numel() for b in model.buffers())
    buffers_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_state_bytes = all_params_bytes + buffers_bytes
    return {
        "n_trainable_params": n_trainable_params,
        "trainable_params_bytes": trainable_params_bytes,
        "n_all_params": n_all_params,
        "all_params_bytes": all_params_bytes,
        "n_buffers": n_buffers,
        "buffers_bytes": buffers_bytes,
        "total_state_bytes": total_state_bytes,
    }

def human_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


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
    if 'pred_semantic_1_1' not in scores_dict:
        return
    pred_sem = torch.argmax(scores_dict['pred_semantic_1_1'], dim=1).data.cpu().numpy()
    curr_index = 0
    for score in pred_sem:
        score = np.moveaxis(score, [0, 1, 2], [0, 2, 1]).reshape(-1).astype(np.uint16)
        score = inv_remap_lut[score].astype(np.uint16)
        input_filename = val_loader.dataset.filepaths['3D_OCCUPANCY'][indices[curr_index]]
        filename, _ = os.path.splitext(os.path.basename(input_filename))
        sequence = os.path.dirname(input_filename).split('/')[-2]
        out_filename = os.path.join(out_path_root, 'sequences', sequence, 'predictions', filename + '.label')
        _create_directory(os.path.dirname(out_filename))
        score.tofile(out_filename)
        if log_each:
            logger.info('=> Sequence {} - File {} saved'.format(sequence, os.path.basename(out_filename)))
        curr_index += 1


@torch.no_grad()
def run_validation(model, val_loader, _cfg, logger, ema, out_path_root=None):
    ema.apply_shadow()  # 应用 EMA 权重
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    model.module.eval()

    inv_remap_lut = val_loader.dataset.get_inv_remap_lut() if hasattr(val_loader.dataset, 'get_inv_remap_lut') \
        else val_loader.dataset.dataset.get_inv_remap_lut()

    for t, (data, indices) in enumerate(val_loader):
        data = dict_to(data, device)
        scores = model(data, stat='test')  # 可能返回 (dict, other...)
        predicts = scores[0]
        for key in predicts:
            predicts[key] = torch.argmax(predicts[key], dim=1).data.cpu().numpy()

        curr_index = 0
        for score in predicts['pred_semantic_1_1']:
            score = np.moveaxis(score, [0, 1, 2], [0, 2, 1]).reshape(-1).astype(np.uint16)
            score = inv_remap_lut[score].astype(np.uint16)
            input_filename = val_loader.dataset.filepaths['3D_OCCUPANCY'][indices[curr_index]]
            filename, extension = os.path.splitext(os.path.basename(input_filename))
            sequence = os.path.dirname(input_filename).split('/')[-2]
            out_filename = os.path.join(out_path_root, 'sequences', sequence, 'predictions', filename + '.label')
            _create_directory(os.path.dirname(out_filename))
            score.tofile(out_filename)
            logger.info('=> Sequence {} - File {} saved'.format(sequence, os.path.basename(out_filename)))
            curr_index += 1


def main():
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
    model = get_model(_cfg, dataset['train'].dataset)

    # DDP & SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 加载权重
    if dist.get_rank() == 0:
        logger.info('=> Loading checkpoint via checkpoint.load(...)')
    model, _, _, epoch = checkpoint.load_double(
        model,
        None,
        None,
        _cfg._dict['STATUS']['RESUME'],
        _cfg._dict['STATUS']['LAST'],
        logger
    )
    if dist.get_rank() == 0:
        logger.info(f'=> Checkpoint loaded (epoch = {epoch})')

    # ★新增：统计并输出参数量（仅 rank 0 打印；统计对象为 model.module，避免 DDP 包装干扰）
    if dist.get_rank() == 0:
        stats = count_params_and_buffers(model.module)
        logger.info(
            "=> Model Parameters (trainable): {:.3f} M ({})".format(
                stats['n_trainable_params'] / 1e6,
                human_bytes(stats['trainable_params_bytes'])
            )
        )
        logger.info(
            "=> Model Parameters (all): {:.3f} M ({})".format(
                stats['n_all_params'] / 1e6,
                human_bytes(stats['all_params_bytes'])
            )
        )
        logger.info(
            "=> Buffers: {:.3f} M ({}) | Total state (params + buffers): {}".format(
                stats['n_buffers'] / 1e6,
                human_bytes(stats['buffers_bytes']),
                human_bytes(stats['total_state_bytes'])
            )
        )

    # 验证
    val_loader = dataset['test']
    ema = EMA(model, 0.999)
    ema.register()

    _ = run_validation(
        model,
        val_loader,
        _cfg,
        logger,
        ema,
        out_path_root="/home/ubuntu/Code/kzw/base_double/SSC_out/predict"
    )


if __name__ == '__main__':
    main()
