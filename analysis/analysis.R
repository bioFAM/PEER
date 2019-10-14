####################
## Load libraries ##
####################

library(MOFA2)

################
## Load model ##
################

file <- "/Users/ricard/data/peer_wrapper/test/model.hdf5"
model <- load_model(file)

####################################
## (Optional) add sample metadata ##
####################################

# Important: 
# (1) the row names in the sample metadata data.frame have to match the sample names in the MOFA model
# (2) the sample name has to contain at least two columns: 
# 		- sample: sample names
#		- group: group name. Only required for multi-group inference. In your case just set the entire column to the same value

# stopifnot(all(samples(model)[[1]]==metadata$sample))
# samples_metadata(model) <- metadata

###############################
## (Optional) Subset factors ##
###############################

# We can remove factors that explain little variance (in this case, we require at least 1%)
r2 <- model@cache$variance_explained$r2_per_factor
factors <- sapply(r2, function(x) x[,1]>0.01)
model <- subset_factors(model, which(apply(factors,1,sum) >= 1))

#############################
## Plot variance explained ##
#############################

# Plot variance explained using individual factors
plot_variance_explained(model, factors="all")
plot_variance_explained(model, factors=c(1,2,3))

# Plot total variance explained using all factors
plot_variance_explained(model, plot_total = TRUE)[[2]]

# Plot variance explained for individual features
features <- c("Rbp4","Ttr","Spink1","Mesp1")
plot_variance_explained_per_feature(model, factors = "all", features = features) # using all factors
plot_variance_explained_per_feature(model, factors = "all", features = features) # using specific factors

########################
## Plot factor values ##
########################

plot_factor(model, 
  factor = 1,
  color_by = "lineage"  # lineage is a column in model@samples.metadata
)

# Other options...
p <- plot_factor(model, 
  factor = 1,
  color_by = "lineage",
  dot_size = 0.2,         # change dot size
  dodge = TRUE,           # dodge points with different colors
  legend = FALSE,         # remove legend
  add_violin = TRUE,      # add violin plots
)


###########################
## Plot feature loadings ##
###########################

# The weights or loadings provide a score for each gene on each factor. 
# Genes with no association with the factor are expected to have values close to zero
# Genes with strong association with the factor are expected to have large absolute values. 
# The sign of the loading indicates the direction of the effect: a positive loading indicates that the feature is more active in the cells with positive factor values, and viceversa.

# Plot the distribution of loadings for Factor 1.
plot_weights(model,
  view = "RNA",
  factor = 1,
  nfeatures = 10,     # Top number of features to highlight
  scale = T           # Scale loadings from -1 to 1
)

# If we are not interested in the directionality of the effect, we can take the absolute value of the loadings. 
# We can also highlight some genes of interest using the argument `manual` to see where in the distribution they lie:
plot_weights(model,
  view = "RNA",
  factor = 1,
  nfeatures = 5,
  manual = list(c("Snai1","Mesp1","Phlda2"), c("Rhox5","Elf5")),
  scale = T,
  abs = T
)

# If you are not interested in the full distribution, but just on the top loadings:
plot_top_weights(model, 
  view = "RNA", 
  factor = 1, 
  nfeatures = 10,
  scale = T, 
  abs = T
)

######################################
## Plot correlation between factors ##
######################################

plot_factor_cor(model)


########################
## Save updated model ##
########################

outfile <- "/Users/ricard/data/peer_wrapper/test/model_updated.rds"
saveRDS(model, outfile)
