import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os 
import numpy as np
from PIL import Image

class p1_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root 
        self.transform = transform
    
        filenames = glob.glob(os.path.join(root, '*.png'))
        #print(filenames)
        for fn in filenames:
            label = fn.split('/', -1)
            label = label[-1]
            label = label.split('_',2)
            label = int(label[0])
            #print(label)
            self.filenames.append((fn, label))

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
        pure_image_fn = image_fn.split('/', -1)
        pure_image_fn = pure_image_fn[-1]
        #print(pure_image_fn)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, pure_image_fn #####
    
    def __len__(self):
        return self.len



if __name__ == '__main__':
    #trainset = p1_dataset(root='hw2_data/p1_data/train_50', transform=transforms.ToTensor())

    augmentation = transforms.Compose([ transforms.RandomHorizontalFlip(0.5),
                                                    transforms.RandomVerticalFlip(0.5),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                    ])

    trainset = p1_dataset(root='hw2_data/p1_data/train_50', transform=augmentation)

    testset = p1_dataset(root='hw2_data/p1_data/val_50', transform=transforms.ToTensor())
    print('# images in trainset:', len(trainset))
    print('# images in testset:', len(testset))

    trainset_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1)
    testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

    dataiter = iter(trainset_loader)
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)

    import matplotlib.pyplot as plt
    import numpy as np
    def imshow(img):
        npimg = img.numpy()
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #plt.show()
    #imshow(torchvision.utils.make_grid(images))
    print('Labels:')
    print(' '.join('%5s' % labels[j] for j in range(16)))