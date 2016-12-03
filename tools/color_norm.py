#!/usr/bin/env python
#
#

#
# Only need the following until matplotlib is removed from the 
# histomics toolkit
#
import matplotlib
matplotlib.use('Agg')

import sys
import numpy
import histomicstk as htk
import h5py


if len(sys.argv) != 3:
	print "Usage: ", sys.argv[0], "<dataset> <standard>"
	exit(-1)


data = h5py.File(sys.argv[1], 'r+')
images = data['images']

print "Normalizing", images.shape[0], "images"

TargetStats = numpy.load(sys.argv[2])

for i in range(images.shape[0]):
	
	dim = images[i].shape[0]
	img = images[i].reshape(dim, dim, 3)

	# MU is the first row of TargetStat, Sigma the second
	#
	norm = htk.ReinhardNorm(img, TargetStats[0], TargetStats[1])

	images[i] = norm.reshape(dim, dim * 3)


print "Normalized", i+1, "images"
