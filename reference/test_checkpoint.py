import torch
from tqdm import tqdm


def read_ssa_sc_keys():
    file_path = '/home/qq/code/gyf/SSA-SC/SSC_out/SSA_SC_SemanticKITTI_CM_REF_2/chkpt/best-metric/weights_epoch_004.pth'
    print('=====================读取模型=====================')
    checkpoint = torch.load(file_path)['model']
    keys = checkpoint.keys()
    print('=====================读取完成=====================')
    SSA_SC_TXT = open('/home/qq/code/gyf/SSA-SC/reference/SSC_SA_KEYS.txt', 'w+')
    for k in tqdm(keys):
        if 'SegNet' in k or 'PPmodel' in k or 'fea_compression' in k :
            SSA_SC_TXT.write(k)
            SSA_SC_TXT.write('\n')
    print('=====================写入完成=====================')

def read_cylinder_keys():
    file_path = '/home/qq/code/gyf/SSA-SC/Pretrain/cyl_sem_1.0x_71_8.pt'
    print('=====================读取模型=====================')
    checkpoint = torch.load(file_path)
    keys = checkpoint.keys()
    print('=====================读取完成=====================')
    SSA_SC_TXT = open('/home/qq/code/gyf/SSA-SC/reference/Cylinder_KEYS.txt', 'w+')
    for k in tqdm(keys):
        SSA_SC_TXT.write(k)
        SSA_SC_TXT.write('\n')
    print('=====================写入完成=====================')
if __name__== '__main__':
    # read_ssa_sc_keys()
    read_ssa_sc_keys()