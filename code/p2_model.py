import torchvision.models as models
import torch
import torch.nn as nn

class fcn(nn.Module):
    def __init__(self , model):
        super(fcn, self).__init__()
        #取掉model最後幾層
        self.model_layer = nn.Sequential(*list(model.children())[:-2])
        self.deconv_32x = nn.Sequential(
            nn.ConvTranspose2d(512, 7, kernel_size=32, stride=32)
        )
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU()
        )
        '''

          
    def forward(self, x):
        x = self.model_layer(x)
        x = self.deconv_32x(x)
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        '''
        return x


if __name__ == '__main__':
    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    vgg16 = models.vgg16(pretrained=True)
    model = fcn(vgg16)
    model = model.to(device) # Remember to move the model to "device"
    print(model)
