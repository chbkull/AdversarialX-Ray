#!/bin/bash

rm data/images/*.png
wget https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz -O images_001.tar.gz
tar -xf images_001.tar.gz -C data
rm images_001.tar.gz