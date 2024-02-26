'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-09-21 17:13:03
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-12-13 20:41:41
FilePath: \HGNN-master_test\datasets\data_helper.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

#import scipy.io as scio
#from os import fstatvfs
from re import L
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
def load_ft(data_dir, feature_name='GVCNN'):
    data_1 = pd.read_csv('data_root\LBPTOP_1.csv',low_memory=False)
    lbls = data_1.iloc[0:,768].to_numpy()
    lbls.astype(np.long)
    if lbls.min() != 0:
        lbls = lbls - lbls.min()
    l=len(lbls)
    nums = np.ones(l)
    nums[int(0.75*l):2386] = 0
    idx=nums
    fts = data_1.iloc[0:,0:768].to_numpy()  
    fts.astype(np.float32)
    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    return fts, lbls, idx_train, idx_test
"""

import scipy.io as scio
import numpy as np

"""
def load_ft(data_dir, feature_name='GVCNN'):
    data = scio.loadmat('data_root/datas.mat')
    lbls = data['Y'].astype(np.long)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()    
    fts = data['X'][1].item().astype(np.float32)
    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    
    return fts, lbls, idx_train, idx_test
"""


# def load_ft(data_dir, feature_name='MVCNN'):
#     data = scio.loadmat(data_dir)
#     lbls = data['Y'].astype(np.long)
#     if lbls.min() == 1:
#         lbls = lbls - 1
#     idx = data['indices'].item()    
#     if feature_name == 'MVCNN':
#         fts = data['X'][0].item().astype(np.float32)
#     elif feature_name == 'GVCNN':
#         fts = data['X'][1].item().astype(np.float32)
#     else:
#         print(f'wrong feature name{feature_name}!')
#         raise IOError
#     idx_train = np.where(idx == 1)[0]
#     idx_test = np.where(idx == 0)[0]
#     return fts, lbls, idx_train, idx_test

"""
def load_ft(data_dir, feature_name='GVCNN'):
    digits = pd.read_csv('data_root/LBPTOP_normal1.csv',low_memory=False,header=None)
    print(digits)
    # 获取数据:data, 标签:target   数据格式[n:样本个数, feature:特征维数]
    x = digits.iloc[1:,0:9].to_numpy()
    x.astype(np.float32)
    y = digits.iloc[1:,777].to_numpy()
    y.astype(np.long)
    # 划分训练集和测试集 7.5:2.5
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    fts = np.concatenate((x_train, x_test), axis=0)
    lbls = np.append(y_train, y_test)
    l=len(lbls)
    nums = np.ones(l)
    nums[int(0.75*l):2373] = 0
    idx=nums
    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    return fts, lbls, idx_train, idx_test
"""