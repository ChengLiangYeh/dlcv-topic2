import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os 
import numpy as np
from PIL import Image
import matplotlib.image as mpimg

class p2_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.masks = None
        self.train_filenames = []
        self.mask_filenames = []
        self.root = root 
        self.transform = transform
        self.filenames = []
        self.mask_not_exist = 0
    
        train_filenames = glob.glob(os.path.join(root, '*.jpg'))
        train_filenames = sorted(train_filenames)
        mask_filenames = glob.glob(os.path.join(root, '*.png'))
        mask_filenames = sorted(mask_filenames)  #window and linux are different. linux need sort
        #print(train_filenames[1])  
        #print(mask_filenames[1])
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
        #print(mask_filenames)

        zipfilename = zip(train_filenames, mask_filenames)
        #print(zipfilename)
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
            #print(labelmap) 
        else:
            labelmap = []

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
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                    ])

    trainset = p2_dataset(root='hw2_data/p2_data/train', transform=augmentation)
    testset = p2_dataset(root='hw2_data/p2_data/validation', transform=transforms.ToTensor())

    print('# images in trainset:', len(trainset))
    print('# images in testset:', len(testset))
    train_batch_size = 1 
    trainset_loader = DataLoader(trainset, train_batch_size, shuffle=False, num_workers=4)
    testset_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

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
    '''超慢...
    def convert_masks_to_labelmap(train_batch_size, masks):
        labelmap = torch.zeros((train_batch_size,512,512), dtype=torch.float32)
        for i in range(train_batch_size):
            for j in range(512):
                for k in range(512):
                    value = masks[i, :, j, k]
                    print(i , j, k)
                    #print(value)
                    if value == [0, 1, 1]:
                        labelmap[i, j, k] = 0
                        print('good')
                    elif value == [1, 1, 0]:
                        labelmap[i, j, k] = 1
                    elif value == [1, 0, 1]:
                        labelmap[i, j, k] = 2
                    elif value == [0, 1, 0]:
                        labelmap[i, j, k] = 3
                    elif value == [0, 0, 1]:
                        labelmap[i, j, k] = 4
                    elif value == [1, 1, 1]:
                        labelmap[i, j, k] = 5
                    elif value == [0, 0, 0]:
                        labelmap[i, j, k] = 6
        return labelmap
    
    labelmap = convert_masks_to_labelmap(train_batch_size, masks)
    print(labelmap)
    '''
    '''成功轉換mask成labelmap
    classes = ['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unknown']
    colormap = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]
    cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i,cm in enumerate(colormap):
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引

    def image2label(im):
        data = np.array(im, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(cm2lbl[idx], dtype='int64')
    
    label_im = Image.open('hw2_data/p2_data/train/0002_mask.png').convert('RGB')
    labelmap = image2label(label_im)
    print(labelmap)
    '''
