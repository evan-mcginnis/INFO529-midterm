#
# I N F O  5 2 9
#
# M I D T E R M  P R O J E C T
#
# This will predict crop yield given a dataset
#


# Toss aside warnings about future versions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import time
import logging
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser("Yield prediction")

parser.add_argument('-m', '--model', action="store", required=True, type=str, help="Model Name")
arguments = parser.parse_args()

# Tensorflow gets a bit picky if you don't have a path qualifier
modelFile = "./" + arguments.model

# Create the saver we will use
saver = tf.train.import_meta_graph("./" + arguments.model + ".meta")

# Load up the previously trained model
print("Loading model {}".format(arguments.model))
with tf.Session() as sess:
    saver.restore(sess, "./" + arguments.model)
print("Loaded")

exit(0)


