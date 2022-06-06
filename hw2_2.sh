#!/bin/bash

wget -O p2-12500-fcn32-miou0.657.pth "https://www.dropbox.com/s/9ypc5cvib6un16h/p2-12500-fcn32-miou0.657.pth?dl=1"

python3 p2_val.py $1 $2
