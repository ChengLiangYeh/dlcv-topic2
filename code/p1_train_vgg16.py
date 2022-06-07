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
from PIL import Image 

def train(model, epoch, save_interval, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95, weight_decay = 0.000001)   ###weight
    #optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode
    
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target, image_fn) in enumerate(trainset_loader):
            #print('target=',target)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            #print('iteration=',iteration) iteration指的就是從頭全部數過來第幾個batch
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
                test(model) #每100個batch test一次測試集

            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('p1-%i.pth' % iteration, model, optimizer)

            iteration += 1

        lr_scheduler.step()
        test(model) # Evaluate at the end of each epoch

    save_checkpoint('p1-%i.pth' % iteration, model, optimizer)

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target, image_fn in testset_loader:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    vgg16 = models.vgg16(pretrained=True)
    model = vgg16
    #print(model)
    #model.classifier._modules['3'] = nn.Linear(4096, 1024)
    model.classifier._modules['6'] = nn.Linear(4096, 50)
    #model.classifier._modules['7'] = nn.ReLU(inplace=True)
    #model.classifier._modules['8'] = nn.Dropout(p = 0.5, inplace=False)
    #model.classifier._modules['9'] = nn.Linear(1000, 50)
    print(model)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    augmentation = transforms.Compose([ transforms.Resize(64, interpolation=2),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomVerticalFlip(0.5),
                                        #transforms.RandomRotation(90, resample=Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    trainset = p1_dataset(root='hw2_data/p1_data/train_50', transform=augmentation)

    testset_augmentation = transforms.Compose([ transforms.Resize(64, interpolation=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])

    testset = p1_dataset(root='hw2_data/p1_data/val_50', transform=testset_augmentation)
    
    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
    train(model, 20, 100, 100)
    
