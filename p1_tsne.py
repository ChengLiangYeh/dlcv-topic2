import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from p1_Dataset import p1_dataset
import torchvision.models as models 
from sklearn import manifold
import matplotlib.pyplot as plt

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        first = 1
        for data, target in testset_loader:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print(output.shape)
            output = output.cpu()
            #print('output=',output.shape)
            target = target.cpu()
            target = target.numpy()

            if first == 1:
                storetensor = torch.empty(1250, 4096)
                #print('uselesstensor=',storetensor)
                storetensor_label = target
                #print(storetensor_label) 
            #print('storetensor=',storetensor.shape)
            storetensor = torch.cat((storetensor, output), 0)
            if first == 0:
                storetensor_label = np.hstack((storetensor_label, target))
                #print('storetensor_label=',storetensor_label.shape)
                #print(storetensor_label)
            first = 0
            #print('storetensor=',storetensor.shape)
        uselesstensor, storetensor = torch.split(storetensor, (1250,2500), dim=0)
        #print(uselesstensor.shape)
        #print('uselesstensor=',uselesstensor)
        #print(storetensor.shape)


        output = storetensor
        target = storetensor_label

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(output)
        print("Our data dimension is {}. Embedded data dimension is {}".format(output.shape[-1], X_tsne.shape[-1]))
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #normalize
        plt.figure(figsize=(8, 8))
        #target = target.cpu()
        #target = target.numpy()
        print(X_tsne.shape)
        print(target.shape)
        for i in range(X_norm.shape[0]):
            color = plt.cm.rainbow(np.linspace(0, 1, 50))
            plt.text(X_norm[i, 0], X_norm[i, 1], str(target[i]), color=color[target[i]], fontdict={'weight': 'bold', 'size': 9})
            #plt.text(X_norm[i, 0], X_norm[i, 1], str(target[i]), color=plt.cm.Set1(target[i]/50), fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig('tsne_result.png')
        plt.show()
        '''
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(output)
            print("Our data dimension is {}. Embedded data dimension is {}".format(output.shape[-1], X_tsne.shape[-1]))
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  #normalize
            plt.figure(figsize=(8, 8))
            target = target.cpu()
            target = target.numpy()
            print(X_tsne.shape)
            print(target.shape)
            for i in range(X_norm.shape[0]):
                color = plt.cm.rainbow(np.linspace(0, 1, 50))
                plt.text(X_norm[i, 0], X_norm[i, 1], str(target[i]), color=color[target[i]], fontdict={'weight': 'bold', 'size': 9})
                #plt.text(X_norm[i, 0], X_norm[i, 1], str(target[i]), color=plt.cm.Set1(target[i]/50), fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.show()
        '''

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


if __name__ == '__main__':
    vgg16 = models.vgg16(pretrained=True)
    model = vgg16
    model.classifier._modules['6'] = nn.Linear(4096, 50)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    #print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #如果是eval結果，寫只是寫好看的 
    load_checkpoint('p1-78%-3-OnTestset-vgg16.pth', model, optimizer)
    
    testset_augmentation = transforms.Compose([ transforms.Resize(64, interpolation=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])

    testset = p1_dataset(root='hw2_data/p1_data/val_50', transform=testset_augmentation)
    testset_loader = DataLoader(testset, batch_size=1250, shuffle=False, num_workers=1)


    #修改
    model.classifier._modules['6'] = nn.Identity()
    print(model)
    test(model)
    # [data_number, 每一筆data有4096維] 
