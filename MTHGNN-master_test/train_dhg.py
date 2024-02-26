import os
import time
import copy
import torch
import torch.optim as optim
import pprint as pp
import dhg
from dhg.models import HGNN
from config import get_config
from dhg import Hypergraph
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dhg.random.set_seed(500)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')


# LBO-TOP特征数据
digits = pd.read_csv('data_root/LBPTOP_SFS_final.csv',low_memory=False,header=None)
# 获取数据:data, 标签:target   数据格式[n:样本个数, feature:特征维数]
x_1 = digits.iloc[0:,3:771].to_numpy()
x_1.astype(np.float32)
y_1 = digits.iloc[0:,771].to_numpy()
y_1.astype(np.long)

# 划分训练集和测试集 7.5:2.5
x_1_train, x_1_test, y_1_train, y_1_test = train_test_split(x_1, y_1, test_size=0.25, random_state=1)
fts = np.concatenate((x_1_train, x_1_test), axis=0)
lbls = np.append(y_1_train, y_1_test)
idx_train = np.array([True]*len(x_1_train) + [False] * len(x_1_test))
idx_test = np.logical_not(idx_train)


# 加载多视图时序特征数据
# with open('data_root/trained_resnet18_fts_max.pkl', 'rb') as fp:
#      data = pkl.load(fp)
# x_2 = np.array(data['fts']) 
# y_2 = np.array(data['lbls'])

with open('data_root/trained_resnet50_fts_max.pkl', 'rb') as fp:
     data = pkl.load(fp)
x_2 = np.array(data['fts']) 
y_2 = np.array(data['lbls'])


f = torch.Tensor(fts)
x = torch.Tensor(x_2)

G = Hypergraph.from_feature_kNN(f,k=2)  
G.add_hyperedges_from_feature_kNN(x,k=23) 

n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# transform data to device
fts = f.to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = G.to(device)
idx_train = torch.Tensor(idx_train).bool().to(device)
idx_test = torch.Tensor(idx_test).bool().to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls[idx])
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / idx.sum()
            epoch_acc = running_corrects.double() / idx.sum()

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20) 
        # print(preds[idx])
        # print(lbls[idx])
        # print("___________________________")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _main():
    print(f"Classification on {cfg['on_dataset']} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {cfg['use_mvcnn_feature']}")
    print(f"use GVCNN feature: {cfg['use_gvcnn_feature']}")
    print(f"use MVCNN feature for structure: {cfg['use_mvcnn_feature_for_structure']}")
    print(f"use GVCNN feature for structure: {cfg['use_gvcnn_feature_for_structure']}")
    print('Configuration -> Start')
    pp.pprint(cfg)
    print('Configuration -> End')

    
    model_ft = HGNN(in_channels=fts.shape[1],
                    num_classes=n_class,
                    hid_channels=cfg['n_hid'],
                    drop_rate=cfg['drop_out'])
    """
    model_ft = HGNN(in_ch=fts.shape[1],
                    n_class=n_class,
                    n_hid=cfg['n_hid'],
                    dropout=cfg['drop_out'])
    """
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=cfg['lr'],
                          weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg('weight_decay'))
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    model_ft = train_model(model_ft, criterion, optimizer, schedular, cfg['max_epoch'], print_freq=cfg['print_freq'])


if __name__ == '__main__':
    _main()


