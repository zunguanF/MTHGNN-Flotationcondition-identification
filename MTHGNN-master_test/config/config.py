'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-09-21 17:13:03
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-10-15 21:38:13
FilePath: \HGNN-master_test\config\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import yaml
import os.path as osp


def get_config(dir='config/config.yaml'):
    # add direction join function when parse the yaml file
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.sep.join(seq)

    # add string concatenation function when parse the yaml file
    def concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!concat', concat)
    with open(dir, 'r') as f:
        cfg = yaml.load(f,Loader=yaml.FullLoader)

    check_dirs(cfg)

    return cfg


def check_dir(folder, mk_dir=True):
    if not osp.exists(folder):
        if mk_dir:
            print(f'making direction {folder}!')
            os.mkdir(folder)
        else:
            raise Exception(f'Not exist direction {folder}')


def check_dirs(cfg):
    check_dir(cfg['data_root'], mk_dir=False)

    check_dir(cfg['result_root'])
    check_dir(cfg['ckpt_folder'])
    check_dir(cfg['result_sub_folder'])
