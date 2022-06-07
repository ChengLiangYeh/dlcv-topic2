#!/bin/bash

wget -O p1-78%-3-OnTestset-vgg16.pth "https://www.dropbox.com/s/ktj0kgf6w5tefos/p1-78%25-3-OnTestset-vgg16.pth?dl=1"
 
python3 p1_val_vgg16.py $1 $2
