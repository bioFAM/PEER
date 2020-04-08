
#########################################
## Script to train a MOFA model from R ##
#########################################

# Important: a part from the MOFA2 R package, you also need the python package mofa2 to be installed
library(MOFA2)

###############
## Load data ##
###############

datafile <- "/Users/ricard/data/peer_wrapper/data.txt.gz"

# The data has to be loaded as a matrix with dimensions (features,samples)
data <- t( read.table(datafile) )

# MOFA is a multi-view factor analysis framework that is a generalisation of PEER
# The data needs to be input as a list of views. If you have a single data modality, the input data 
# corresponds to a list with a single element.
data <- list(data)

########################
## Create MOFA object ##
########################

object <- suppressMessages(create_mofa(data))

# check dimensionalities
print(object)

#########################
## Define data options ##
#########################

data_opts <- get_default_data_options(object)

##########################
## Define model options ##
##########################

model_opts <- get_default_model_options(object)

# use ARD prior for the factors? (please do not edit this)
model_opts$ard_factors <- FALSE

# number of factors
model_opts$num_factors <- 10


#############################
## Define training options ##
#############################

train_opts <- get_default_training_options(object)

# maximum number of iterations
train_opts$maxiter <- 2000

# fast, medium, slow
train_opts$convergence_mode <- "fast"

# initial iteration to start evaluating convergence using the ELBO (recommended >1)
train_opts$startELBO <- 1

# frequency of evaluation of ELBO (recommended >1)
train_opts$freqELBO <- 1

# use GPU (needs cupy installed and working)
train_opts$gpu_mode <- FALSE

# verbose output?
train_opts$verbose <- FALSE

# random seed
train_opts$seed <- 1

#############################
## Prepare the MOFA object ##
#############################

object <- prepare_mofa(
  object = object, 
  data_options = data_opts,
  model_options = model_opts,
  training_options = train_opts
)

##############
## Run MOFA ##
##############

# The output is an hdf5 file that can be loaded using the function load_model(file)
outfile = "/Users/ricard/data/peer_wrapper/model.hdf5"
model <- run_mofa(object, outfile=outfile)
