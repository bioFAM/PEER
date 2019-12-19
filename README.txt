###################################
## Run PEER using MOFA framework ##
###################################

###############################
## Installation instructions ##
###############################

# Python
pip install mofapy2

# R
devtools::install_github("bioFAM/MOFA2/MOFA2", build_opts = c("--no-resave-data --no-build-vignettes"))


#############
## Folders ##
#############

run: scripts to train the model
analysis: scripts to analyse the model
