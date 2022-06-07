#!/bin/bash
wget -O p2-final-24734-best-miou-0.7305.pth "https://www.dropbox.com/s/oilz9tvwys0ih10/p2-final-24734-best-miou-0.7305.pth?dl=1"
python3 p2_val_final.py $1 $2
