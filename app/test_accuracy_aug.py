import os
import sys
import random
import subprocess
import re
import numpy
import time
start = time.time()

# Define parameters
TOTAL_SLIDES = 88
TRAIN_SLIDES = 80
TEST_SLIDES = 8
REPEAT = 5

train_net_path = "./train_net"
test_net_path = "./test_net"

accs = []
for i in range(REPEAT):

    # Create Folds
    train_slides = range(TRAIN_SLIDES)
    random.shuffle(train_slides)
	
    test_slides = range(TOTAL_SLIDES-TEST_SLIDES, TOTAL_SLIDES)
    random.shuffle(test_slides)

    # Call program
    train_set = ",".join(str(e) for e in train_slides)
    test_set = ",".join(str(e) for e in test_slides) 
    
    # Print datasets
    print "Trainning on:"
    print train_set

    print "Testing on:"
    print test_set

    # Remove previous saved model
    if os.path.isfile("./cell_net.caffemodel"):
        os.remove("./cell_net.caffemodel")

    # Call processes
    subprocess.call([train_net_path, train_set])
    output = subprocess.check_output([test_net_path, test_set])

    # Get accuracy
    print "Computing accuracy for model {0}".format(i)
    pred = []
    for k in range(4):
        with open('pred_{0}.txt'.format(k)) as f:
            data = f.read().split()
            pred.append([int(x[0]) for x in data])

    pred = numpy.sum(numpy.array(pred), axis=0)
    pred = numpy.array(pred >= 2).astype(int)
    size = len(pred)
   
    targets = []
    with open('pred_{0}.txt'.format(0)) as f:
            data = f.read().split()
            targets.append([int(x[2]) for x in data])
    targets = numpy.ravel(targets)

    pred = numpy.sum(numpy.array(pred), axis=0)
    pred = numpy.array(pred >= 2).astype(int)

    hits = 0
    for i, e in enumerate(pred):
        if int(e) == int(targets[i]):
            hits += 1

    print "Predictions: "
    print pred

    print "Targets: "
    print targets

    print "Hits: " + str(hits)
    print "Total: " + str(size)

    acc = float(hits)/size
    accs.append(acc)
    
    for k in range(4):

        # Remove pred
        os.remove("./pred_{0}.txt".format(k))
    print "Accs so far"
    print accs

# Show results
print accs
print numpy.mean(accs)
print numpy.median(accs)

print "Elapsed time:"
print time.time() - start
