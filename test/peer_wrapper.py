#!/g/stegle/ricard/anaconda/envs/mofa2/bin/python
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores
#SBATCH --mem 7G                   # memory pool for all cores
#SBATCH -o slurm.%N.%j.out          # STDOUT
#SBATCH -e slurm.%N.%j.err          # STDERR
#SBATCH --mail-type=END,FAIL        # notifications for job done & fail
#SBATCH --mail-user=ricard@ebi.ac.uk

# sbatch -p gpu --gres gpu:1 run.py

from mofapy2.run.entry_point import entry_point
import numpy as np
import pandas as pd

###############
## Load data ##
###############

# datafile = "/hps/nobackup2/research/stegle/users/ricard/peer/data/FullFreeze_Corrected_iPSC_TPM2_20180626_hqS.txt.gz"
datafile = "/g/stegle/ricard/peer/data/FullFreeze_Corrected_iPSC_TPM2_20180626_hqS.txt.gz"

# The data has to be loaded as a pandas dataframe or as a numpy matrix with dimensions (samples,features)
data = pd.read_csv(datafile, header=0, sep='\t', index_col=0)

# Define likelihoods: non-gaussian likelihoods are implemented (poisson and bernoulli), but by default we use gaussian.
lik = ["gaussian"]

###########################
## Initialise MOFA model ##
###########################

# initialise the entry point
ent = entry_point()

# Set data options
ent.set_data_options(likelihoods=lik)

# Set data
ent.set_data_matrix([[data]]) # do not modify this nested list

# Set model options
# - factors: number of factors
# - spikeslab_weights: use spike-and-slab sparsity on the loading?
# - ard_weights: use ARD prior on the loadings (please do not edit this)
ent.set_model_options(factors=100, spikeslab_weights=False, ard_weights=False, likelihoods=lik)

# Set training options
# - iter: maximum number of iterations
# - convergence_mode: fast, medium, slow
# - startELBO: initial iteration to start evaluating convergence using the ELBO (recommended >1)
# - elbofreq: frequency of evaluation of ELBO (recommended >1)
# - gpu_mode: use GPU (need cupy installed and working)
# - verbose: verbosity
# - seed
ent.set_train_options(iter=5000, convergence_mode="medium", startELBO=200, elbofreq=5, gpu_mode=True, verbose=False, seed=1)

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
outfile = "/hps/nobackup2/research/stegle/users/ricard/peer/hdf5/without_sparsity/model_medium.hdf5"
ent.save(outfile)
