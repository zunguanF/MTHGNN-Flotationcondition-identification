import time
import pickle as pkl
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import copy

device = "cuda:0"
root = Path('datasets/')


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class MyDataset(Dataset):
    def __init__(self, img_list, y_list, transform):
        self.trans = transform
        self.img_list = img_list
        self.y_list = y_list
    
    def __len__(self):
        return len(self.y_list)
    
    def __getitem__(self, index):
        img_pathes, y = self.img_list[index], self.y_list[index]
        imgs = []
        for img_path in img_pathes:
            img = Image.open(img_path)
            img = self.trans(img)
            imgs.append(img)
        imgs = torch.stack(imgs) # 时间序列
        return imgs, y

class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(pretrained=True)
        self.net = nn.Sequential(*list(net.children())[:-1])
        self.cls = nn.Linear(512, 4)
    
    def forward(self, x, ret_ft=False):
        bz, n_views = x.shape[0], x.shape[1]
        # merge view
        x = x.view(bz*n_views, x.shape[2], x.shape[3], x.shape[4])
        ft = self.net(x)
        # extract view
        ft = ft.view(bz, n_views, ft.shape[1])
        # max view pooling
        ft = ft.max(dim=1)[0]
        if ret_ft:
            return ft
        else:
            return self.cls(ft)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def extract_fts(dataloader, model):
    model = model.eval()
    all_fts, all_lbls = [], []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        fts = model(inputs, ret_ft=True)
        fts = fts.cpu().detach().numpy()
        lbls = labels.cpu().detach().numpy()
        all_fts.append(fts)
        all_lbls.append(lbls)
    all_fts = np.concatenate(all_fts)
    all_lbls = np.concatenate(all_lbls)
    return all_fts, all_lbls


if __name__ == "__main__":

    data_list, y_list = [], []
    for lbl_idx, cate_root in enumerate(root.glob('*')):
        for instance_root in cate_root.glob('*'):
            cur = list(instance_root.glob('*.jpg'))
            assert len(cur) == 12
            data_list.append(cur)
            y_list.append(lbl_idx)

    x_train, x_test, y_train, y_test = train_test_split(data_list, y_list, test_size=0.25, random_state=1)

    train_data = MyDataset(x_train, y_train, data_transforms["train"])
    test_data = MyDataset(x_test, y_test, data_transforms["test"])
    dataloaders = {
        "train": DataLoader(train_data, 64, True, num_workers=4),
        "test": DataLoader(test_data, 64, False, num_workers=4)
    }
    dataset_sizes = {
        "train": len(x_train),
        "test": len(x_test)
    }

    # train network
    model_ft = FeatureNet().to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)

    # extract feature
    print(f"extracting features...")
    dataloaders = {
        "train": DataLoader(train_data, 64, False, num_workers=4),
        "test": DataLoader(test_data, 64, False, num_workers=4)
    }
    train_fts, train_lbls = extract_fts(dataloaders["train"], model_ft)
    test_fts, test_lbls = extract_fts(dataloaders["test"], model_ft)
    all_fts = np.concatenate([train_fts, test_fts])
    all_lbls = np.concatenate([train_lbls, test_lbls])
    train_idx = np.array([True] * len(train_fts) + [False] * len(test_fts))
    test_idx = np.logical_not(train_idx)

    data = {
        'fts': all_fts,
        'lbls': all_lbls,
        'train_idx': train_idx,
        'test_idx': test_idx
    }

    with open('feature/trained_resnet18_fts_max.pkl', 'wb') as fp:
        pkl.dump(data, fp)

