import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os 
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import random

class p2_dataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.images = None
        self.masks = None
        self.train_filenames = []
        self.mask_filenames = []
        self.root = root 
        self.transform = transform
        self.filenames = []
        self.mode = mode
        self.mask_not_exist = 0
    
        train_filenames = glob.glob(os.path.join(root, '*.jpg'))
        train_filenames = sorted(train_filenames)
        mask_filenames = glob.glob(os.path.join(root, '*.png'))
        mask_filenames = sorted(mask_filenames)
        #print(mask_filenames)
        if mask_filenames == []:
            self.mask_not_exist = 1
            for i in train_filenames:
                #print(i)
                mask_filename = i.split("/", -1)
                mask_filename = mask_filename[-1]
                mask_filename = mask_filename.split("_", -1)
                mask_filename = mask_filename[0]
                mask_filename = root + '/' + mask_filename + '_mask.png'
                #print(mask_filename)
                mask_filenames.append(mask_filename)

        zipfilename = zip(train_filenames, mask_filenames)
        for fn in zipfilename:
            self.filenames.append((fn))
        self.len = len(self.filenames)


    def __getitem__(self, index):
        image_fn, mask_fn = self.filenames[index]
        image = Image.open(image_fn)
        #print(image_fn)
        if self.mask_not_exist == 0:
            mask = Image.open(mask_fn)
        else:
            mask = []
        #
        classes = ['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unknown']
        colormap = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]
        cm2lbl = np.zeros(256**3)
        for i,cm in enumerate(colormap):
            cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

        def image2label(im): #用在下面
            data = np.array(im, dtype='int32')
            idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
            return np.array(cm2lbl[idx], dtype='int64')
        if self.mask_not_exist == 0:
            label_im = Image.open(mask_fn).convert('RGB')
            #print(label_im.shape)'Image' object has no attribute 'shape'
            labelmap = image2label(label_im)
            #print(type(labelmap))
            #print(labelmap.shape)
        else:
            labelmap = [] 
        
        def augment(image, mask, labelmap):
            choice = ['yes', 'no']
            up_down_ischoice = random.choice(choice)
            left_right_ischoice = random.choice(choice)
            rotate_minus90_ischoice = random.choice(choice)
            rotate_minus180_ischoice = random.choice(choice)
            #print('up_down_ischoice= ',up_down_ischoice)
            #print('left_right_ischoice= ',left_right_ischoice)
            #print('rotate_minus90_ischoice= ',rotate_minus90_ischoice)
            
            if up_down_ischoice == 'yes':
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                labelmap = np.flip(labelmap, 0).copy()

            if left_right_ischoice == 'yes':
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                labelmap = np.flip(labelmap, 1).copy()
            
            if rotate_minus90_ischoice == 'yes':
                image = image.transpose(Image.ROTATE_90)
                mask = mask.transpose(Image.ROTATE_90)
                labelmap = np.rot90(labelmap).copy()
            
            if rotate_minus180_ischoice == 'yes':
                image = image.transpose(Image.ROTATE_180)
                mask = mask.transpose(Image.ROTATE_180)
                labelmap = np.rot90(labelmap,2).copy()

            return image, mask, labelmap


        if self.mode == 'train':
            image, mask, labelmap = augment(image, mask, labelmap)
        

        if self.transform is not None:
            image = self.transform(image)
            if self.mask_not_exist == 0:
                mask = self.transform(mask)
            else:
                mask = []
            
            #labelmap = self.transform(labelmap) -> 只能是pil image
            
        return image, mask, labelmap, mask_fn
    
    def __len__(self):
        return self.len



if __name__ == '__main__':
    #trainset = p1_dataset(root='hw2_data/p1_data/train_50', transform=transforms.ToTensor())

    augmentation = transforms.Compose([ #transforms.RandomHorizontalFlip(0.5),
                                        #transforms.RandomVerticalFlip(0.5),
                                        #transforms.RandomRotation(15),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                    ])

    trainset = p2_dataset(root='hw2_data/p2_data/train', transform=augmentation, mode='train')
    testset = p2_dataset(root='hw2_data/p2_data/validation', transform=transforms.ToTensor(), mode='test')

    print('# images in trainset:', len(trainset))
    print('# images in testset:', len(testset))
    train_batch_size = 2
    trainset_loader = DataLoader(trainset, train_batch_size, shuffle=False, num_workers=0) #num_workers會平行取更多張喔!!(如果忘記意思就回來設成4看看)
    testset_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    dataiter = iter(trainset_loader)
    images, masks, labelmap, mask_fn = dataiter.next() #labelmap is a single channel pic converted from masks(3 channels pic)
    print(labelmap)
    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Mask tensor in each batch:', masks.shape, masks.dtype)
    print('labelmap in each batch:', labelmap.shape, labelmap.dtype)

    import matplotlib.pyplot as plt
    import numpy as np
    def imshow(img):
        npimg = img.numpy()
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #plt.show()
    #imshow(torchvision.utils.make_grid(images))
    #imshow(torchvision.utils.make_grid(masks))
