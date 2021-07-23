## R packages
library(tidyverse)
library(here)
library(Rtsne)
library(ggplot2)
library(MASS)
library(patchwork)

## PCA projection
calculate_pca <- function(feature_dataset){
  pcaY_cal <- prcomp(feature_dataset, center = TRUE, scale = TRUE)
  
  PCAresults <- data.frame(PC1 = pcaY_cal$x[, 1], 
                           PC2 = pcaY_cal$x[, 2], 
                           PC3 = pcaY_cal$x[, 3],
                           PC4 = pcaY_cal$x[, 4],
                           PC5 = pcaY_cal$x[, 5],
                           PC6 = pcaY_cal$x[, 6],
                           PC7 = pcaY_cal$x[, 7],
                           PC8 = pcaY_cal$x[, 8])
  return(list(prcomp_out =pcaY_cal,pca_components = PCAresults))
}
pca_projection <- function(prcomp_out, data_to_project){
  
  PCA <- scale(data_to_project, prcomp_out$center, prcomp_out$scale) %*% prcomp_out$rotation
  pca_projected <- data.frame(PC1=PCA[,1], PC2=PCA[,2], PC3=PCA[,3], PC4=PCA[,4], PC5=PCA[,5], PC6=PCA[,6], PC7=PCA[,7], PC8=PCA[,8]) 
  return(pca_projected)
  
}
pca_summary <- function(feature_dataset){
  pcaY_cal <- prcomp(feature_dataset, center = TRUE, scale = TRUE)
  
  return(summary(pcaY_cal))
}


## Swedish -----
data_new <- read.csv("data_all_with_label_with_species.csv", header = TRUE)
head(data_new)

features <- data_new[, c(3:10,12:53)] # remove Outlying_polar and Outlying_contour
pca_ref_calc <- calculate_pca(features)
# combine features and PCs' into a one dataframe
data_new$PC1 <- pca_ref_calc$pca_components$PC1
data_new$PC2 <- pca_ref_calc$pca_components$PC2
data_new$PC3 <- pca_ref_calc$pca_components$PC3
data_new$PC4 <- pca_ref_calc$pca_components$PC4
data_new$PC5 <- pca_ref_calc$pca_components$PC5
data_new$PC6 <- pca_ref_calc$pca_components$PC6
data_new$PC7 <- pca_ref_calc$pca_components$PC7
data_new$PC8 <- pca_ref_calc$pca_components$PC8

p11 <- ggplot(data_new, aes(x=PC1, y=PC2, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA1 Vs PCA2 by Actual Species") + xlab("PCA1") + ylab("PCA2") + theme(aspect.ratio = 1) 
p12 <- ggplot(data_new, aes(x=PC1, y=PC3, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA1 Vs PCA3 by Actual Species") + xlab("PCA1") + ylab("PCA3") + theme(aspect.ratio = 1) 
p13 <- ggplot(data_new, aes(x=PC1, y=PC4, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA1 Vs PCA4 by Actual Species") + xlab("PCA1") + ylab("PCA4") + theme(aspect.ratio = 1) 
p14 <- ggplot(data_new, aes(x=PC1, y=PC5, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA1 Vs PCA5 by Actual Species") + xlab("PCA1") + ylab("PCA5") + theme(aspect.ratio = 1) 
p15 <- ggplot(data_new, aes(x=PC2, y=PC3, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA2 Vs PCA3 by Actual Species") + xlab("PCA2") + ylab("PCA3") + theme(aspect.ratio = 1) 
p16 <- ggplot(data_new, aes(x=PC2, y=PC4, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA2 Vs PCA4 by Actual Species") + xlab("PCA2") + ylab("PCA4") + theme(aspect.ratio = 1) 
p17 <- ggplot(data_new, aes(x=PC2, y=PC5, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA2 Vs PCA5 by Actual Species") + xlab("PCA2") + ylab("PCA5") + theme(aspect.ratio = 1) 
p18 <- ggplot(data_new, aes(x=PC3, y=PC4, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA3 Vs PCA4 by Actual Species") + xlab("PCA3") + ylab("PCA4") + theme(aspect.ratio = 1) 
p19 <- ggplot(data_new, aes(x=PC3, y=PC5, col=Species)) + geom_point(alpha = 0.5) +coord_equal()+ scale_colour_manual(values = c("#a6611a","#d7191c","#e66101","#1b9e77","#3182bd","#d95f02","#7570b3","#fc8d92","#31a354","#7b3294","#d01c8b","#8da0cb","#b2df8a")) + ggtitle("PCA3 Vs PCA5 by Actual Species") + xlab("PCA3") + ylab("PCA5") + theme(aspect.ratio = 1) 

p1h1 <- p11 + theme(legend.position = "none") + theme(plot.title = element_blank())
p2h1 <- p12 + theme(legend.position = "none") + theme(plot.title = element_blank())
p3h1 <- p13 + theme(plot.title = element_blank()) + theme(legend.position = "none")
p4h1 <- p14 + theme(plot.title = element_blank()) + theme(legend.position = "none")
p5h1 <- p15 + theme(plot.title = element_blank()) + theme(legend.position = "none")
p6h1 <- p16 + theme(legend.position = "none") + theme(plot.title = element_blank())
p7h1 <- p17 + theme(plot.title = element_blank()) + theme(legend.position = "none")
p8h1 <- p18 + theme(plot.title = element_blank()) + theme(legend.position = "none")
p9h1 <- p19 + theme(plot.title = element_blank())
p1h1 + p2h1 + p3h1 + p4h1 + p5h1 + p6h1 + p7h1 + p8h1 + p9h1 + plot_annotation(
  title = 'PCA with Actual Species',
  tag_levels = 'A'
) & theme(plot.tag = element_text(size = 5))
