rm(list = ls())
getwd()
setwd("~/GitHub/ParEvol/")

library(PLNmodels)
library(data.table)

# generate network for tenaillon dataset 
df.tenaillon <- read.table("data/Tenaillon_et_al/gene_by_pop_delta.txt", 
                           header = TRUE, sep = "\t", row.names = 1)

matrix.mult <- as.matrix(df.tenaillon)
mult.models <- PLNnetwork(matrix.mult ~ 1)
#mult.models.bic <- mult.models$getBestModel("BIC")
#write.table(as.matrix(mult.models.bic$latent_network()), file = "data/Tenaillon_et_al/network_bic.txt", sep = "\t")

mult.models.stars <- mult.models$getBestModel("StARS")
write.table(as.matrix(mult.models.stars$latent_network()), file = "data/Tenaillon_et_al/network_stars.txt", sep = "\t")
