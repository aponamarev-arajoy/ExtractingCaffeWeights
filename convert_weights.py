#!/usr/bin/env python
"""
convert_weights is a tool for conversion of Caffe weight blobs into numpy tensors.
Created 7/21/17 at 11:55 AM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import sys
import cPickle
import os
sys.path.insert(0,'/opt/caffe/python')
import numpy as np
import caffe
print "Caffe version used: ", caffe.__version__

proto = "model/_trained_COCO/pose_deploy.prototxt" #"./pose_deploy.prototxt"
caffemodel = "model/_trained_COCO/pose_iter_440000.caffemodel" #"./pose_iter_440000.caffemodel"

export_path = "caffe_weights.p"

net = None

caffe_layers = {}

def check_file(p):
    assert os.path.isfile(p), "Error: provided path ({}) doesn't exist.".format(p)
    return p

def caffe_weights(name):
    return net.params[name][0].data

def caffe_bias(name):
    return net.params[name][1].data

def caffe2tf_filter(name):
  f = caffe_weights(name)
  return f.transpose((2, 3, 1, 0))

def caffe2tf_conv_blob(name):
  blob = net.blobs[name].data[0]
  return blob.transpose((1, 2, 0))

def caffe2tf_1d_blob(name):
  blob = net.blobs[name].data[0]
  return blob

if __name__ == "__main__":

    # Initialize a net
    net = caffe.Net(check_file(proto), check_file(caffemodel), caffe.TEST)

    # Extract all parameters
    #params = {k: map(data, v) for k, v in net.params.items()}

    # Aggregate layers
    for name in net.params.keys():
        weights, bias = None, None
        try:
            weights = caffe2tf_filter(name)
        except:
            print name + " weights weren't found."

        try:
            bias = caffe_bias(name)
        except:
            print name + " biases weren't found."
        caffe_layers[name] = {'w': weights, 'b': bias}

    print "Finished converting the data."

    with open(export_path, 'wb') as f:
        cPickle.dump(caffe_layers, f)

    print  "Saved data."



