#!/usr/bin/env sh

/home/nelson/caffe/build/tools/caffe test -model /home/nelson/CellNet/caffe_model/cnn_test.prototxt -weights /home/nelson/CellNet/caffe_model/snapshot/_iter_50000.caffemodel -gpu 0 -iterations 1
