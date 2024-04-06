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
    'nE_tsp_20/ref_svoA2080_YJ_20240107T170532': 'icml_nE_tsp20_ref',
    'nE_tsp_20/svd_k5_rel_svoA2080_YJ_20240113T180835': 'icml_nE_tsp20_relK5',
    'nE_tsp_20/svd_k5_org_svoA2080_YJ_20240113T180838': 'icml_nE_tsp20_orgK5',
    'nE_tsp_20/svd_k20_rel_svoA2080_YJ_20240120T161641': 'icml_nE_tsp20_relK20',
    'nE_tsp_20/svd_k20_org_svoA2080_YJ_20240120T161734': 'icml_nE_tsp20_orgK20',
    'nE_tsp_20/svd_onlyD_svo2080_YJ_20240113T180902': 'icml_nE_tsp20_orgK20_onlyD',
    'nE_tsp_20/E1_rel_svo2080_YJ_20240115T081911': 'icml_nE_tsp20_relE1',
    'nE_tsp_20/E_org_svo2080_YJ_20240102T143811': 'icml_nE_tsp20_orgE1',
    'nE_tsp_20/E3_rel_svo2080_YJ_20240115T081857': 'icml_nE_tsp20_relE3',
    'nE_tsp_20/E3_org_svo2080_YJ_20240115T081857': 'icml_nE_tsp20_orgE3',
    'nE_tsp_20/E_org_k5_rel_svoA5k_YJ_20240102T143421': 'icml_nE_tsp20_relK5_orgE1',
    'nE_tsp_20/E3_org_k5_rel_svoA5k_YJ_20240107T170222': 'icml_nE_tsp20_relK5_orgE3',
    'nE_tsp_20/E6_org_k5_rel_pomo_16_ultra_svoA5k_YJ_20240109T003927': 'icml_nE_tsp20_relK5_orgE6_ultra',
    # nE_tsp_50
    'nE_tsp_50/ref_svo2080_YJ_20240107T215237': 'icml_nE_tsp50_ref',
    'nE_tsp_50/svd_k5_rel_svoA2080_YJ/_20240113T181612': 'icml_nE_tsp50_relK5',
    'nE_tsp_50/svd_k5_org_svoA2080_YJ_20240113T181634': 'icml_nE_tsp50_orgK5',
    'nE_tsp_50/svd_k20_rel_svoA2080_YJ_20240126T125549': 'icml_nE_tsp50_relK20',
    'nE_tsp_50/svd_k20_org_svoA2080_YJ_20240126T125449': 'icml_nE_tsp50_orgK20',
    'nE_tsp_50/E1_rel_svo2080_YJ_20240115T083100': 'icml_nE_tsp50_relE1',
    'nE_tsp_50/E_org_svo2080_YJ_20240102T144612': 'icml_nE_tsp50_orgE1',
    'nE_tsp_50/E3_rel_svo2080_YJ_20240115T083142': 'icml_nE_tsp50_relE3',
    'nE_tsp_50/E3_org_svo2080_YJ_20240115T083123': 'icml_nE_tsp50_orgE3',
    'nE_tsp_50/E_org_k5_rel_svo2080_YJ_20240102T144612': 'icml_nE_tsp50_relK5_orgE1',
    'nE_tsp_50/E3_org_k5_rel_svo2080_YJ_20240115T083604': 'icml_nE_tsp50_relK5_orgE3',
    'nE_tsp_50/E6_org_k5_rel_pomo_16_ultra_svoA5k_YJ_20240113T124127': 'icml_nE_tsp50_relK5_orgE6_ultra',
    # nE_tsp_100
    'nE_tsp_100/ref_svoA5k_YJ_20240113T182539': 'icml_nE_tsp100_ref',
    'nE_tsp_100/svd_k5_rel_svoA5k_YJ_20240116T194530': 'icml_nE_tsp100_relK5',
    'nE_tsp_100/svd_k5_org_svoA5k_YJ_20240116T194955': 'icml_nE_tsp100_orgK5',
    'nE_tsp_100/svd_k20_rel_svoA5k_YJ_20240129T180036': 'icml_nE_tsp100_relK20',
    # 'nE_tsp_100/svd_k20_org_svoA5k_YJ_20240129T180130': 'icml_nE_tsp100_orgK20',
    'nE_tsp_100/E1_rel_svoA5K_YJ_20240118T215544': 'icml_nE_tsp100_relE1',
    'nE_tsp_100/E1_org_svoA5K_YJ_20240129T180510': 'icml_nE_tsp100_orgE1',
    'nE_tsp_100/E3_rel_svoA5K_YJ_20240118T215702': 'icml_nE_tsp100_relE3',
    'nE_tsp_100/E3_org_svoA5K_YJ_20240118T214848': 'icml_nE_tsp100_orgE3',
    'nE_tsp_100/E1_org_k5_rel_svoA5K_YJ_20240129T180948': 'icml_nE_tsp100_relK5_orgE1',
    'nE_tsp_100/E3_org_k5_rel_svoA5K_YJ_20240118T215008': 'icml_nE_tsp100_relK5_orgE3',
    'nE_tsp_100/E6_org_k5_rel_pomo_16_ultra_svoA40_YJ_20240118T220521': 'icml_nE_tsp100_relK5_orgE6_ultra',
    # nE_tsp_20_rS
    # 'nE_tsp_20_rS/new_k0_rel_svo2080_YJ_20231124T184120': 'Nov28_nE_tsp20_C_ref',
    # 'nE_tsp_20_rS/new_k3_rel_svo2080_YJ_20231124T165952': 'Nov28_nE_tsp20_C_relK3',
    # 'nE_tsp_20_rS/new_k3_org_svo2080_YJ_20231124T170024': 'Nov28_nE_tsp20_C_orgK3',
    # 'nE_tsp_20_rS/new_k20_org_onlyD_svo2080_YJ_20231124T172113': 'Nov28_nE_tsp20_C_orgK20_onlyD',
    # 'nE_tsp_20_rS/E_rel_svo1080_YJ_20231126T022133': 'Nov28_nE_tsp20_C_relE',
    # 'nE_tsp_20_rS/E_rel_NrS_svo1080_YJ_20231126T022154': 'Nov28_nE_tsp20_C_NrS_relE',
    # 'nE_tsp_20_rS/E_org_svo1080_YJ_20231126T022154': 'Nov28_nE_tsp20_C_orgE',
    # 'nE_tsp_20_rS/E_org_NrS_svo1080_YJ_20231126T113832': 'Nov28_nE_tsp20_C_NrS_orgE',
    # nE_tsp_50_rS
    'nE_tsp_50_rS/ref_svo1080_YJ_20240119T142810': 'icml_nE_tsp50_C_ref',
    'nE_tsp_50_rS/E3_org_k5_rel_svo1080_YJ_20240129T181411': 'icml_nE_tsp50_C_relK5_orgE3',
    'nE_tsp_50_rS/rS_E3_org_k0_rel_pomo16_svoA5k_YJ_20240130T115905': 'icml_nE_tsp50_C_orgE3_rS',
    # nE_tsp_100_rS
    # ...
    # 'nE_tsp_100_rS/E_rel_svoA5k_YJ_20231126T023130': 'Nov28_nE_tsp100_C_relE',
    # 'nE_tsp_100_rS/E_rel_NrS_svoA5k_YJ_20231126T023146': 'Nov28_nE_tsp100_C_NrS_relE',
    # 'nE_tsp_100_rS/E_org_svoA5k_YJ_20231126T023137': 'Nov28_nE_tsp100_C_orgE',
    # 'nE_tsp_100_rS/E_org_NrS_svoA5k_YJ_20231126T023158': 'Nov28_nE_tsp100_C_NrS_orgE',
    # nE_cvrp_20
    'nE_cvrp_20/am_ref_svo2080_YJ_20240118T195719': 'icml_nE_cvrp20_ref',
    'nE_cvrp_20/E3_org_k5_rel_svo2080_YJ_20240118T195653': 'icml_nE_cvrp20_relK5_orgE3',
    # nE_cvrp_50
    'nE_cvrp_50/am_ref_svo2080_YJ_20240118T195749': 'icml_nE_cvrp50_ref',
    'nE_cvrp_50/E3_org_k5_rel_svo2080_YJ_20240129T182046': 'icml_nE_cvrp50_relK5_orgE3',
    # nE_cvrp_100
    'nE_cvrp_100/am_ref_svoA5k_YJ_20240130T114126': 'icml_nE_cvrp100_ref',
    'nE_cvrp_100/E3_org_k5_rel_svoA5k_YJ_20240129T182422': 'icml_nE_cvrp100_relK5_orgE3',
}

from_dir = os.path.join(curr_dir, 'outputs')
to_dir = os.path.join(curr_dir, 'pretrained/Jan26-icml')

for from_name, to_name in copy_dict.items():
    from_path = os.path.join(from_dir, from_name)
    to_path = os.path.join(to_dir, to_name)
    copy_trained_nets(from_path, to_path)
    # logger.info(f'Copied {from_path} to {to_path}')
logger.success('Done')