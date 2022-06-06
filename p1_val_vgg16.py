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
import pandas as pd
import sys#### 

def test(model, output_csv_folder_root):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    time = 1
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target, image_fn in testset_loader:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #####
            #print(image_fn)
            image_fn_arr = np.array(image_fn)
            #print(image_fn_arr)
            #print(image_fn_arr.shape)
            image_fn_arr = image_fn_arr.reshape([1250, 1])
            #print(image_fn_arr.shape)
            #print(pred.shape)
            #print(image_fn_arr)
            pred_copy = pred.cpu()
            pred_copy = pred_copy.numpy()
            #print(pred_copy.shape)
            #print(pred_copy)
            pred_copy = np.hstack((image_fn_arr,pred_copy))
            #print(pred_copy)
            if time == 1 :
                pd_data1 = pd.DataFrame(pred_copy, columns = ['image_id','label'])
                #print('pd_data1=',pd_data1)
                time += 1
            elif time == 2:
                pd_data2 = pd.DataFrame(pred_copy, columns = ['image_id','label'])
                #print('pd_data2=',pd_data2)
            #####
            correct += pred.eq(target.view_as(pred)).sum().item()
    pd_total_data = pd.concat( [pd_data1, pd_data2] ) #####
    #print(pd_total_data) #####
    pd_total_data.to_csv(output_csv_folder_root + '/test_pred.csv',index=False) #####
    print('save output csv file')
    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))

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
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #如果是eval結果，寫只是寫好看的 
    load_checkpoint('p1-78%-3-OnTestset-vgg16.pth', model, optimizer)
    
    testset_augmentation = transforms.Compose([ transforms.Resize(64, interpolation=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])
    testing_image_directory_root = sys.argv[1]
    print('$1=',testing_image_directory_root)
    output_csv_folder_root = sys.argv[2]
    print('$2=',output_csv_folder_root)
    testset = p1_dataset(root=testing_image_directory_root, transform=testset_augmentation)
    testset_loader = DataLoader(testset, batch_size=1250, shuffle=False, num_workers=1)
    test(model, output_csv_folder_root)