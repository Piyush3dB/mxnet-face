#!/usr/bin/env sh 

rm -rf ./predict.txt
#align_data_path=/home/work/data/Face/LFW/lfw-align
align_data_path=/home/piyush/Downloads/temp/lfw-deepfunneled
model_prefix=model/lightened_cnn/lightened_cnn
epoch=166
# evaluate on lfw
python lfw.py --lfw-align $align_data_path --model-prefix $model_prefix --epoch $epoch 
