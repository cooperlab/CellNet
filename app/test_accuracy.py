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
    acc = re.search(r'Accuracy: ([^\s]+)', output)
    accs.append(float(acc.group(0).split(" ")[1]))
    
    # Remove pipe
    os.remove("./pipe0")

# Show results
print accs
print numpy.mean(accs)
print numpy.median(accs)

print "Elapsed time:"
print time.time() - start
