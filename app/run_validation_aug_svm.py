import os
import sys
import random
import subprocess
import re
import numpy
import time
from sklearn import svm

start = time.time()

# Define parameters
NUMBER_OF_SLIDES = 80
NUMBER_OF_FOLDS = 10

train_net_path = "./train_net"
test_net_path = "./test_net"

# Create Folds
fold_size = NUMBER_OF_SLIDES/NUMBER_OF_FOLDS
slides = range(NUMBER_OF_SLIDES)
random.shuffle(slides)

# Call program
accs = []
for i in range(NUMBER_OF_FOLDS):

    print "Setting up dataset for model {0}".format(i)
    # Split up datasets
    valid_slides = slides[i*fold_size:(i+1)*fold_size]
    valid_set = ",".join(str(e) for e in valid_slides)
    train_slides = list(set(slides)-set(valid_slides))
    random.shuffle(train_slides)
    train_set = ",".join(str(e) for e in train_slides[:-fold_size])
    train_slides_svm = train_slides[-fold_size:]
    train_set_svm = ",".join(str(e) for e in train_slides_svm)

    # Print datasets
    print "Trainning CNN on:"
    print train_set

    print "Training SVM on:"
    print train_set_svm

    print "Testing on:"
    print valid_set

    # Remove previous saved model
    if os.path.isfile("./cell_net.caffemodel"):
    	os.remove("./cell_net.caffemodel")

    # Call processes
    print "Training CNN for model {0}".format(i)
    subprocess.call([train_net_path, train_set])

    print "Training SVM for model {0}".format(i)
    subprocess.check_output([test_net_path, train_set_svm])
    pred = []
    for k in range(4):
        with open('pred_{0}.txt'.format(k)) as f:
            data = f.read().split()
            pred.append([int(x[0]) for x in data])

    targets = []
    with open('pred_{0}.txt'.format(0)) as f:
        data = f.read().split()
        targets.append([int(x[2]) for x in data])

    # Fit SkLearn    
    clf = svm.SVC()
    pred = numpy.transpose(pred)    
    targets = numpy.ravel(targets)
    clf.fit(pred, targets)

    # Remove txts
    for k in range(4):
        os.remove("./pred_{0}.txt".format(k))

    print "Testing for model {0}".format(i)
    subprocess.check_output([test_net_path, valid_set])

    # Predict using CNN
    print "Computing accuracy for model {0}".format(i)
    pred = []
    for k in range(4):
        with open('pred_{0}.txt'.format(k)) as f:
            data = f.read().split()
            pred.append([int(x[0]) for x in data])
   
    # Predict using SVM
    pred = numpy.transpose(pred)
    svm_pred = clf.predict(pred)
    size = len(svm_pred)    
    targets = []
    with open('pred_{0}.txt'.format(0)) as f:
            data = f.read().split()
            targets.append([int(x[2]) for x in data])
    targets = numpy.ravel(targets)

    hits = 0
    for i, e in enumerate(svm_pred):
        if int(e) == int(targets[i]):
            hits += 1

    print "CNNs Predictions: "
    print pred

    print "SVM Predictions: "    
    print svm_pred

    print "Targets: "
    print targets

    print "Hits: " + str(hits)
    print "Total: " + str(size)

    # Remove txt
    for k in range(4):
        os.remove("./pred_{0}.txt".format(k))

    print "Accs so far"
    print accs

# Show results
print accs
print numpy.mean(accs)
print numpy.median(accs)
print "Elapsed time:"
print time.time() - start
