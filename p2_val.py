import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from p2_Dataset import p2_dataset
import torchvision.models as models
from PIL import Image
from p2_model import fcn
import matplotlib.pyplot as plt
import sys 

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, masks, target, mask_fn in testset_loader:
            #print(mask_fn)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            if masks == [] and target == []:
                data = data.to(device)
            else:
                data, target = data.to(device), target.to(device)
            output = model(data)

            colormap = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]
            cm = np.array(colormap).astype('uint8')
            pred = output.max(1)[1].squeeze().cpu().data.numpy()
            pred = cm[pred]
            #print(pred.shape)

            if len(pred.shape) == 3:
                pred_mask = pred[:, :, :]
                pred_filename = mask_fn[0]
                #print(pred_filename)
                pred_filename = pred_filename.split("/", -1)
                #print(pred_filename)
                pred_filename = output_image_directory_root + '/' + pred_filename[-1]
                #print(pred_filename)
                plt.imsave(pred_filename, pred_mask)
            else:   
                for i in range((pred.shape[0])):
                    #print(i)
                    pred_mask = pred[i, :, :, :]
                    #print(pred_mask.shape)
                    #print(pred_mask)
                    #print('.......................................................')
                    #plt.imshow(pred_mask)
                    #plt.show()
                    pred_filename = mask_fn[i]
                    #print(type(pred_filename))
                    #print(pred_filename)
                    pred_filename = pred_filename.split("/", -1)
                    #print(pred_filename)
                    pred_filename = output_image_directory_root + '/' +pred_filename[-1]
                    #print(pred_filename)
                    plt.imsave(pred_filename, pred_mask)
            if target != []:
                test_loss += criterion(output, target).item() # sum up batch loss
    if target != []:
        test_loss /= len(testset_loader.dataset)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))

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
    model = fcn(vgg16)
    model = model.to(device) # Remember to move the model to "device"
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #如果是eval結果，寫只是寫好看的
    load_checkpoint('p2-12500-fcn32-miou0.657.pth', model, optimizer)
    testset_augmentation = transforms.Compose([ #transforms.Resize(64, interpolation=2),
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])
    
    testing_image_directory_root = sys.argv[1]
    print('$1=',testing_image_directory_root)
    output_image_directory_root = sys.argv[2]
    print('$2=',output_image_directory_root)
    testset = p2_dataset(root=testing_image_directory_root, transform=testset_augmentation) #root='hw2_data/p2_data/validation'
    testset_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
    test(model)