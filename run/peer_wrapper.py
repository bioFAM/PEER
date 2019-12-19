######################################################
## Template script to train a MOFA model via Python ##
######################################################

from mofapy2.run.entry_point import entry_point
import numpy as np
import pandas as pd

###############
## Load data ##
###############

datafile = "/Users/ricard/data/peer_wrapper/data.txt.gz"

# The data has to be loaded as a pandas dataframe or as a numpy matrix with dimensions (samples,features)
data = pd.read_csv(datafile, header=0, sep='\t')

###########################
## Initialise MOFA model ##
###########################

# initialise the entry point
ent = entry_point()

# Set data
# MOFA is a multi-view and multi-group inference framework. 
# If usig only a single view and a single group (as in PEER), the data needs to be embedded into a nested list
ent.set_data_matrix([[data]])

# Set model options
# - factors: number of factors
# - spikeslab_weights: use spike-and-slab sparsity on the loading?
# - ard_weights: use ARD prior on the loadings (please do not edit this)
ent.set_model_options(factors=10, spikeslab_weights=False, ard_weights=False)

# Set training options
# - iter: maximum number of iterations
# - convergence_mode: fast, medium, slow
# - startELBO: initial iteration to start evaluating convergence using the ELBO (recommended >1)
# - freqELBO: frequency of evaluation of ELBO (recommended >1)
# - gpu_mode: use GPU (need cupy installed and working)
# - verbose: verbosity
# - seed
ent.set_train_options(iter=1000, convergence_mode="fast", startELBO=1, freqELBO=1, gpu_mode=False, verbose=False, seed=42)

# (Optional) Set stochastic inference options
# - batch_size: batch size, has to be a multiple of 5 (0.05, 0.10, ..., 0.25, maximum is 0.5)
# - learning_rate: initial learning rate (recommended between 0.25 and 0.5)
# - forgetting_rate: decreasing learning rate (recommended from 0.25 to 0.75)
# ent.set_stochastic_options(batch_size=.5, learning_rate=0.75, forgetting_rate=0.5)

####################################
## Build and train the MOFA model ##
####################################

ent.build()
ent.run()

####################
## Save the model ##
####################

# The output is an hdf5 file that can be subsequently loaded into R (no python package for the downstream analysis is available yet)
outfile = "/Users/ricard/data/peer_wrapper/model.hdf5"
ent.save(outfile)
