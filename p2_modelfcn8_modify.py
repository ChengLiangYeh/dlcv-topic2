import torchvision.models as models
import torch
import torch.nn as nn

class fcn8(nn.Module):
    def __init__(self , model):
        super(fcn8, self).__init__()
        #取掉model最後幾層
        '''
        self.model_layer = nn.Sequential(*list(model.children())[:-2])
        self.deconv_4x = nn.Sequential(
            nn.ConvTranspose2d(512, 1, kernel_size=4, stride=4)
        )
        self.model_layer_from_pool4 = model.features[0:24]
        self.deconv_2x = nn.Sequential(
            nn.ConvTranspose2d(512, 1, kernel_size=2, stride=2)
        )
        self.model_layer_from_pool3 = model.features[0:17]
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )

        self.deconv_8x = nn.Sequential(
            nn.ConvTranspose2d(1, 7, kernel_size=8, stride=8)
        )
        '''
        self.model_layer_from_pool3 = model.features[0:17]

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.middlelayer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.deconv_2x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )

        self.lastlayer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.deconv_4x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=4)
        )

        self.deconv_2x_2layer = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )

        self.deconv_2x_tochannel_7 = nn.Sequential(
            nn.ConvTranspose2d(512, 7, kernel_size=2, stride=2)
        )


    def forward(self, x):
        '''
        x1 = x
        x2 = x
        x3 = x

        x1 = self.model_layer(x1)
        x1 = self.deconv_4x(x1)

        x2 = self.model_layer_from_pool4(x2)
        x2 = self.deconv_2x(x2)

        x3 = self.model_layer_from_pool3(x3)
        x3 = self.conv1_1(x3)

        finalx = x1 + x2 + x3 
        finalx = self.deconv_8x(finalx)

        return finalx
        '''
        
        x_main = self.model_layer_from_pool3(x)
        x_branch1 = self.conv1_1(x_main)
        x_main = self.middlelayer(x_main)
        x_branch2 = self.deconv_2x(x_main)
        x_main = self.lastlayer(x_main)
        x_branch3 = self.deconv_4x(x_main)
        x_final = x_branch1 + x_branch2 + x_branch3
        x_final = self.deconv_2x_2layer(x_final)
        x_final = self.deconv_2x_tochannel_7(x_final)
        return x_final


if __name__ == '__main__':
    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    vgg16 = models.vgg16(pretrained=True)
    model = fcn8(vgg16)
    model = model.to(device) # Remember to move the model to "device"
    print(model)