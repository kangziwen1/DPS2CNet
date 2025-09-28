import numpy as np
import torch

from networks.models.LMSCNet import LMSCNet
from networks.models.LMSCNet_SS import LMSCNet_SS

from networks.models.SSA_SC_MREF import SSA_SC_MREF
from networks.models.SSA_SC_MREF_GN import SSA_SC_MREF_GN
from networks.models.SSA_SC_MREF_OC import SSA_SC_MREF_OC
from networks.models.SSA_SC_REF import SSA_SC_REF
from networks.models.SSA_SC_REF_ACode import SSA_SC_REF_AC
from networks.models.SSCNet_full import SSCNet_full
from networks.models.SSCNet import SSCNet
from networks.models.LMSCNet_SSP import LMSCNet_SSP
from networks.models.BEV_UNet import BEV_UNet
from networks.models.SSA_SC import SSA_SC
from networks.models.SSA_SC_CrossModal import SSA_SC_CM


def get_class_weights(_cfg):
    seg_num_per_class = _cfg._dict['seg_labelweights']
    seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
    seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
    # tensor([inf, 2.0484, 11.4097, 9.5813, 5.6957, 5.2598, 9.5177, 13.7308,
    #         18.2139, 1.2552, 3.0203, 1.4083, 4.2135, 1.2740, 1.6765, 1.0000,
    #         3.6458, 1.5434, 4.7286, 7.4162])
    return seg_labelweights


def get_model(_cfg, dataset):
    nbr_classes = dataset.nbr_classes
    grid_dimensions = dataset.grid_dimensions
    class_frequencies = dataset.class_frequencies

    selected_model = _cfg._dict['MODEL']['TYPE']

    # LMSCNet ----------------------------------------------------------------------------------------------------------
    if selected_model == 'LMSCNet':
        model = LMSCNet(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
    # ------------------------------------------------------------------------------------------------------------------

    # LMSCNet_SS -------------------------------------------------------------------------------------------------------
    elif selected_model == 'LMSCNet_SS':
        model = LMSCNet_SS(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
    # ------------------------------------------------------------------------------------------------------------------

    # SSCNet_full ------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSCNet_full':
        model = SSCNet_full(class_num=nbr_classes)
    # ------------------------------------------------------------------------------------------------------------------

    # SSCNet -----------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSCNet':
        model = SSCNet(class_num=nbr_classes)
    # ------------------------------------------------------------------------------------------------------------------

    # LMSCNet_SSP ------------------------------------------------------------------------------------------------------
    elif selected_model == 'LMSCNet_SSP':
        model = LMSCNet_SSP(class_num=nbr_classes, input_dimensions=grid_dimensions,
                            class_frequencies=class_frequencies, pt_model='pointnet', fea_dim=7, pt_pooling='max',
                            kernal_size=3, out_pt_fea_dim=512, fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------

    # BEV_Unet ---------------------------------------------------------------------------------------------------------
    elif selected_model == 'BEV_Unet':
        model = BEV_UNet(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies,
                         pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                         fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------

    # SSA_SC --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC':
        model = SSA_SC(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies,
                       pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                       fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------

    # SSA_SC_CM --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC_CM':
        model = SSA_SC_CM(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies,
                          pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                          fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------

    # SSA_SC_REF --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC_REF':
        model = SSA_SC_REF(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies,
                           pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                           fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------

    # SSA_SC_REF_AC --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC_REF_AC':
        model = SSA_SC_REF_AC(class_num=nbr_classes, input_dimensions=grid_dimensions,
                              class_frequencies=class_frequencies,
                              pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                              fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------

    # SSA_SC_REF_AC --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC_MREF':
        model = SSA_SC_MREF(class_num=nbr_classes, input_dimensions=grid_dimensions,
                               class_frequencies=class_frequencies,
                               pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                               fea_compre=32)
    # ----------------------------------------------------------------------------------------------------------------

    # SSA_SC_REF_AC --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC_MREF_OC':
        model = SSA_SC_MREF_OC(class_num=nbr_classes, input_dimensions=grid_dimensions,
                            class_frequencies=class_frequencies,
                            pt_model='pointnet', fea_dim=9, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                            fea_compre=32)
    # ----------------------------------------------------------------------------------------------------------------

    # SSA_SC_MREF_GN --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC_MREF_GN':
        model = SSA_SC_MREF_GN(class_num=nbr_classes, input_dimensions=grid_dimensions,
                              class_frequencies=class_frequencies,
                              pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512,
                              fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------
    else:
        assert False, 'Wrong model selected'

    return model

    # model = ptBEVnet(BEV_model, pt_model='pointnet', grid_size=grid_dimensions, fea_dim=7, out_pt_fea_dim=512, kernal_size=3, fea_compre=32)
