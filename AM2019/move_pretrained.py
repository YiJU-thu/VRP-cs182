import os, sys
from loguru import logger

curr_dir = os.path.dirname(__file__)
util_dir = os.path.join(curr_dir, '..', 'utils_project')
if util_dir not in sys.path:
    sys.path.append(util_dir)

from clear_optimizer_states import copy_trained_nets
import torch

copy_dict = {
    # nE_tsp_20
    'tsp_20/k0_rel_svo1080_YJ_20231122T194348': 'Nov28_nE_tsp20_ref',
    'nE_tsp_20/new_k3_rel_svo2080_YJ_20231124T165906': 'Nov28_nE_tsp20_relK3',
    'nE_tsp_20/new_k3_org_svo2080_YJ_20231124T165906': 'Nov28_nE_tsp20_orgK3',
    'nE_tsp_20/new_k10_rel_svo2080_YJ_20231124T214648': 'Nov28_nE_tsp20_relK10',
    'nE_tsp_20/new_k10_org_svo2080_YJ_20231124T214720': 'Nov28_nE_tsp20_orgK10',
    'nE_tsp_20/new_k20_rel_svo2080_YJ_20231124T214554': 'Nov28_nE_tsp20_relK20',
    'nE_tsp_20/new_k20_org_svo2080_YJ_20231124T214616': 'Nov28_nE_tsp20_orgK20',
    'nE_tsp_20/new_k20_org_onlyD_svo2080_YJ_20231124T171947': 'Nov28_nE_tsp20_orgK20_onlyD',
    'nE_tsp_20/E_rel_svo2080_YJ_20231125T172153': 'Nov28_nE_tsp20_relE',
    'nE_tsp_20/E_org_svo2080_YJ_20231125T172437': 'Nov28_nE_tsp20_orgE',
    'nE_tsp_20/E_rel_k5_rel_svo2080_YJ_20231126T022438': 'Nov28_nE_tsp20_relK5_relE',
    'nE_tsp_20/E_org_k5_rel_svo2080_YJ_20231126T022552': 'Nov28_nE_tsp20_relK5_orgE',
    # nE_tsp_50
    'nE_tsp_50/test_k0_rel_svo2080_YJ_20231123T021356': 'Nov28_nE_tsp50_ref',
    'nE_tsp_50/new_k5_rel_fast_svo2080_YJ_20231124T191454': 'Nov28_nE_tsp50_relK5',
    'nE_tsp_50/new_k10_rel_svo2080_YJ_20231124T215107': 'Nov28_nE_tsp50_relK10',
    'nE_tsp_50/E_rel_svo2080_YJ_20231125T172513': 'Nov28_nE_tsp50_relE',
    'nE_tsp_50/E_org_svo2080_YJ_20231125T172521': 'Nov28_nE_tsp50_orgE',
    'nE_tsp_50/new_k5_org_svo2080_YJ_20231126T023720': 'Nov28_nE_tsp50_orgK5',
    'nE_tsp_50/new_k10_org_svo2080_YJ_20231126T023711': 'Nov28_nE_tsp50_orgK10',
    'nE_tsp_50/E_rel_k5_rel_svo2080_YJ_20231126T023548': 'Nov28_nE_tsp50_relK5_relE',
    'nE_tsp_50/E_org_k5_rel_svo2080_YJ_20231126T023556': 'Nov28_nE_tsp50_relK5_orgE',
    # nE_tsp_100
    'nE_tsp_100/test_k0_rel_svoA5k_YJ_20231123T021820': 'Nov28_nE_tsp100_ref',
    'nE_tsp_100/new_k5_rel_fast_svoA5k_YJ_20231124T191507': 'Nov28_nE_tsp100_relK5',
    'nE_tsp_100/E_rel_svoA5k_YJ_20231125T173135': 'Nov28_nE_tsp100_relE',
    'nE_tsp_100/E_org_svoA5k_YJ_20231125T173038': 'Nov28_nE_tsp100_orgE',
    'nE_tsp_100/new_k5_org_svoA5k_YJ_20231126T023922': 'Nov28_nE_tsp100_orgK5',
    # 'nE_tsp_100/new_k20_rel_svoA5k_YJ_20231124T215227': 'Nov28_nE_tsp100_relK20',
    # 'nE_tsp_100/new_k20_org_svoA5k_YJ_20231126T023922': 'Nov28_nE_tsp100_orgK20',
    'nE_tsp_100/E_rel_k5_rel_svoA5k_YJ_20231126T024839': 'Nov28_nE_tsp100_relK5_relE',
    'nE_tsp_100/E_org_k5_rel_svoA5k_YJ_20231126T024853': 'Nov28_nE_tsp100_relK5_orgE',
    # nE_tsp_20_rS
    'nE_tsp_20_rS/new_k0_rel_svo2080_YJ_20231124T184120': 'Nov28_nE_tsp20_C_ref',
    'nE_tsp_20_rS/new_k3_rel_svo2080_YJ_20231124T165952': 'Nov28_nE_tsp20_C_relK3',
    'nE_tsp_20_rS/new_k3_org_svo2080_YJ_20231124T170024': 'Nov28_nE_tsp20_C_orgK3',
    'nE_tsp_20_rS/new_k20_org_onlyD_svo2080_YJ_20231124T172113': 'Nov28_nE_tsp20_C_orgK20_onlyD',
    'nE_tsp_20_rS/E_rel_svo1080_YJ_20231126T022133': 'Nov28_nE_tsp20_C_relE',
    'nE_tsp_20_rS/E_rel_NrS_svo1080_YJ_20231126T022154': 'Nov28_nE_tsp20_C_NrS_relE',
    'nE_tsp_20_rS/E_org_svo1080_YJ_20231126T022154': 'Nov28_nE_tsp20_C_orgE',
    'nE_tsp_20_rS/E_org_NrS_svo1080_YJ_20231126T113832': 'Nov28_nE_tsp20_C_NrS_orgE',
    # nE_tsp_50_rS
    # 'nE_tsp_50_rS/...': 'Nov28_nE_tsp50_C_ref',
    # 'nE_tsp_50_rS/...': 'Nov28_nE_tsp50_C_relK3',
    # 'nE_tsp_50_rS/...': 'Nov28_nE_tsp50_C_orgK3',
    # 'nE_tsp_50_rS/...': 'Nov28_nE_tsp50_C_orgK20_onlyD',
    'nE_tsp_50_rS/E_rel_svo2080_YJ_20231126T022644': 'Nov28_nE_tsp50_C_relE',
    'nE_tsp_50_rS/E_rel_NrS_svo2080_YJ_20231126T022701': 'Nov28_nE_tsp50_C_NrS_relE',
    'nE_tsp_50_rS/E_org_svo2080_YJ_20231126T022647': 'Nov28_nE_tsp50_C_orgE',
    'nE_tsp_50_rS/E_org_NrS_svo2080_YJ_20231126T022714': 'Nov28_nE_tsp50_C_NrS_orgE',
    # nE_tsp_100_rS
    # ...
    'nE_tsp_100_rS/E_rel_svoA5k_YJ_20231126T023130': 'Nov28_nE_tsp100_C_relE',
    'nE_tsp_100_rS/E_rel_NrS_svoA5k_YJ_20231126T023146': 'Nov28_nE_tsp100_C_NrS_relE',
    'nE_tsp_100_rS/E_org_svoA5k_YJ_20231126T023137': 'Nov28_nE_tsp100_C_orgE',
    'nE_tsp_100_rS/E_org_NrS_svoA5k_YJ_20231126T023158': 'Nov28_nE_tsp100_C_NrS_orgE',
}

from_dir = os.path.join(curr_dir, 'outputs')
to_dir = os.path.join(curr_dir, 'pretrained/Nov28-cs182')

for from_name, to_name in copy_dict.items():
    from_path = os.path.join(from_dir, from_name)
    to_path = os.path.join(to_dir, to_name)
    copy_trained_nets(from_path, to_path)
    # logger.info(f'Copied {from_path} to {to_path}')
logger.success('Done')