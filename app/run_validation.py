import os
import sys
import random
import subprocess
import re
import numpy
import time
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

    # Split up datasets
    valid_slides = slides[i*fold_size:(i+1)*fold_size]
    valid_set = ",".join(str(e) for e in valid_slides)
    train_slides = list(set(slides)-set(valid_slides))
    random.shuffle(train_slides)
    train_set = ",".join(str(e) for e in train_slides) 
    
    # Print datasets
    print "Trainning on:"
    print train_set

    print "Testing on:"
    print valid_set

    # Remove previous saved model
    if os.path.isfile("./cell_net.caffemodel"):
    	os.remove("./cell_net.caffemodel")

    # Call processes
    subprocess.call([train_net_path, train_set])
    output = subprocess.check_output([test_net_path, valid_set])

    # Get accuracy
    acc = re.search(r'Accuracy: ([^\s]+)', output)
    accs.append(float(acc.group(0).split(" ")[1]))
    
    # Remove pipe
    os.remove("./pipe0")

# Show results
print accs
print numpy.mean(accs)
print "Elapsed time:"
print time.time() - start
