import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from p2_Dataset_final import p2_dataset #經過data augmentation之後的dataset
import torchvision.models as models
from PIL import Image
from p2_modelfcn8_modify import fcn8  #跟上一代p2_trainfcn8_modify.py用同一個model，原因是之前在寫程式時是先精進model之後才精進dataset，因此model目前已經算OK了

def train(model, epoch, save_interval, log_interval):
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0001)   
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  
    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode
    
    iteration = 0
    best_train_loss = 0
    count_no_progress = 0
    EarlyStop = 0
    for ep in range(epoch):
        for batch_idx, (data, masks, target, mask_fn) in enumerate(trainset_loader):
            #print('target=',target)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            #print(target.shape)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #print(output.shape)
            loss = criterion(output, target)
            #print(loss.item())
            if best_train_loss == 0:
                best_train_loss = loss.item()
            if best_train_loss > loss.item():
                best_train_loss = loss.item()
                count_no_progress += 1
                #print('count_no_progress= ',count_no_progress)
            else:
                count_no_progress = 0
            if count_no_progress == 5:
                print('Early Stop!!!')
                earlyStop = 1
                break
                
            loss.backward()
            optimizer.step()
            
            #print('iteration=',iteration) iteration指的就是從頭全部數過來第幾個batch
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
                test(model) #每100個batch test一次測試集

            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('p2-final-%i.pth' % iteration, model, optimizer)

            iteration += 1
        if EarlyStop == 1:
            break

        #lr_scheduler.step()
        #test(model) # Evaluate at the end of each epoch

    save_checkpoint('p2-final-%i.pth' % iteration, model, optimizer)

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, masks, target, mask_fn in testset_loader:
            #print(mask_fn)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

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

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    vgg16 = models.vgg16(pretrained=True)
    model = fcn8(vgg16)
    model = model.to(device) # Remember to move the model to "device"
    print(model)

    augmentation = transforms.Compose([ #transforms.Resize(64, interpolation=2),
                                        #transforms.RandomHorizontalFlip(0.5),
                                        #transforms.RandomVerticalFlip(0.5),
                                        #transforms.RandomRotation(90, resample=Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  
                                        ])
    trainset = p2_dataset(root='hw2_data/p2_data/train', transform=augmentation, mode='train')

    testset_augmentation = transforms.Compose([ #transforms.Resize(64, interpolation=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  
                                                ])
    testset = p2_dataset(root='hw2_data/p2_data/validation', transform=testset_augmentation, mode='test')
    
    train_batch_size = 6
    trainset_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
    train(model, 100, 166, 166)