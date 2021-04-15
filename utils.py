import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=True, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

def get_loaders(dataset, label_class, batch_size, datapath=None):
    if dataset in ['cifar10', 'fashion']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    elif dataset.lower() == 'cub':
        if datapath is None:
            raise RuntimeError('Please provide a path to the Caltech Birds 200 dataset with --datapath=')
        trainset = CUB_Dataset(path=datapath, train=True)
        testset  = CUB_Dataset(path=datapath, train=False)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()

def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)



import os, PIL.Image

class CUB_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, train=False):
        path       = os.path.expanduser(path)
        imagefiles = open(os.path.join(path, 'images.txt')).read().split('\n')[:-1]
        imagefiles = [os.path.join(path, 'images', line.split(' ')[1]) for line in imagefiles]
        labels     = open(os.path.join(path, 'image_class_labels.txt')).read().split('\n')[:-1]
        labels     = [int(line.split(' ')[1]) for line in labels]
        split      = open(os.path.join(path, 'train_test_split.txt')).read().split('\n')[:-1]
        split      = [int(line.split(' ')[1]) for line in split]
        
        #defining the first 20 classes as normal as in paper, appendix C
        normal_classes = np.unique(labels)[:20]
        
        if train:
            #indices of training set as defined by the dataset authors
            ixs = [i for i in range(len(imagefiles)) if split[i]==1]
            #remove anormal images
            ixs = [i for i in ixs if labels[i] in normal_classes]
        else:
            #indices of test set as defined by the dataset authors
            ixs = [i for i in range(len(imagefiles)) if split[i]==0]
        
        self.imagefiles = [imagefiles[i] for i in ixs]
        self.labels     = [labels[i] for i in ixs]
        
        if train:
            #re-using labels as above for cifar10/fmnist
            #targets of trainset are not used anyway
            self.targets    = self.labels
        else:
            #1 if not in normal classes, 0 otherwise
            self.targets    = [int(l not in normal_classes) for l in self.labels]
        
        self.transform  = transform_color
    
    def __len__(self):
        return len(self.imagefiles)
    
    def __getitem__(self, i):
        imagefile = self.imagefiles[i]
        image     = PIL.Image.open(imagefile).convert('RGB')
        image     = self.transform(image)
        target    = self.targets[i]
        return image, target
