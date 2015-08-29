import os
import sys
import random
import subprocess
import re
import numpy

# Define parameters
NUMBER_OF_SLIDES = 88
NUMBER_OF_FOLDS = 10
HOLD_OUT_NUMBER = 1
train_net_path = "./train_net"
test_net_path = "./test_net"

# Create Folds
fold_size = NUMBER_OF_SLIDES/(NUMBER_OF_FOLDS+HOLD_OUT_NUMBER)
slides = range(NUMBER_OF_SLIDES)
random.shuffle(slides)

# Hold slides out
holdout_slide = slides[-fold_size:]
slides = slides[:-fold_size]

# Call program
accs = []
print slides
for i in range(NUMBER_OF_FOLDS):

    # Split up datasets
    valid_slides = slides[i*fold_size:(i+1)*fold_size]
    valid_set = ",".join(str(e) for e in valid_slides)
    train_set = ",".join(str(e) for e in list(set(slides)-set(valid_slides)))

    # Call processes
    subprocess.call([train_net_path, train_set])
    output = subprocess.check_output([test_net_path, valid_set])

    # Get accuracy
    acc = re.search(r'Accuracy: ([^\s]+)', output)
    accs.append(acc.group(0))

# Show results
print accs
print numpy.mean(accs)