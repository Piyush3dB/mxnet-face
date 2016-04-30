#!/usr/bin/env python

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import os
import sys
import argparse,logging
import find_mxnet
import mxnet as mx
import cv2
from lightened_cnn import lightened_cnn_b_feature
import pdb as pdb



ctx = mx.gpu(0)





def mltp(x):
    s = 1
    for i in range(0,len(x)):
        s *= x[i]

    return s



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default="./pairs.txt",
                        help='Location of the LFW pairs file from http://vis-www.cs.umass.edu/lfw/pairs.txt')
    parser.add_argument('--lfw-align', type=str, default="./lfw-align",
                        help='The directory of lfw-align, which contains the aligned lfw images')
    parser.add_argument('--suffix', type=str, default="png",
                        help='The type of image')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--model-prefix', default='model/lightened_cnn/lightened_cnn',
                        help='The trained model to get feature')
    parser.add_argument('--epoch', type=int, default=165,
                        help='The epoch number of model')
    parser.add_argument('--predict-file', type=str, default='./predict.txt',
                        help='The file which contains similarity distance of every pair image given in pairs.txt')
    args = parser.parse_args()
    logging.info(args)


    ctx = mx.model.load_checkpoint('lightened_cnn', 166)

    pretrained_model = mx.model.FeedForward.load("lightened_cnn", 166)

    arg_names    = pretrained_model.symbol.list_arguments()
    output_names = pretrained_model.symbol.list_outputs()

    # Infer arg and output shapes
    input_size = (1, 1, 128, 128)
    arg_shapes, output_shapes, aux_shapes = pretrained_model.symbol.infer_shape(data=input_size)
    
    



    # Display
    nWtot = 0
    print "Layer Params... "
    for i in range(0,len(arg_shapes)):
        nW = mltp(arg_shapes[i])
        nWtot += nW
        print '%35s -> %25s = %10s' % (arg_names[i], str(arg_shapes[i]), str(nW))
    
    print 'Total weights   = %d' % (nWtot)
    print 'Total weights M = %f' % (nWtot/1000000.)


    pdb.set_trace()


    # Display
    print "Layer Outputs... "
    for i in xrange(len(output_names)):
        print '%35s -> %25s = %10s' % (output_names[i], output_shapes[i], str(mltp(output_shapes[i])))








if __name__ == '__main__':
    main()