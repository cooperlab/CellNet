import h5py
import numpy as np

dataset_path = "/home/mnalisn/testsets/LGG-Endothelial-combined-fixed.h5"
f = h5py.File(dataset_path)

labels = f['labels']
slideIdx = f['slideIdx']
slides = f['slides']

labels_count = np.zeros((2, len(slides)))

for i, l in enumerate(labels):
    
    if int(l) == -1:
        labels_count[0][slideIdx[i]] += 1
    else:
        labels_count[1][slideIdx[i]] += 1
       
for i, s in enumerate(slides):
    print "Slide Name: " + s
    print "SlideIdx: " + str(i)
    print "Negative Elements Count: " + str(int(labels_count[0][i]))
    print "Positive Elements Count: " + str(int(labels_count[1][i])) 
    print "Total Elements Count: " + str(int(labels_count[0][i] + labels_count[1][i]))
    print "\n"


