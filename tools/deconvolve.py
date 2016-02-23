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


if len(sys.argv) != 2:
	print "Usage: ", sys.argv[0], "<dataset>"
	exit(-1)


W = numpy.array([[0.650, 0.072, 0],
				 [0.704, 0.990, 0],
				 [0.286, 0.105, 0]])


data = h5py.File(sys.argv[1], 'r+')
images = data['images']

print "Processing", images.shape[0], "images"

for i in range(images.shape[0]):

	print "Image", i
	
	dim = images[i].shape[0]
	img = images[i].reshape(dim, dim, 3)

	deconv = htk.ColorDeconvolution(img, W)

	images[i] = deconv.Stains.reshape(dim, dim * 3)


data.close()
