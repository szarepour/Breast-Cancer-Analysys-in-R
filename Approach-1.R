# load libraries
# install.packages("ggplot2")            # Packages need to be installed only once
# install.packages("GGally")
library(ggplot2)                     # Load ggplot2 package
library(GGally)                      # Load GGally package
library(tidyverse)
library(dplyr) 
library(MVN)
library(corrplot)
library(caret)
library(ggfortify)
library(factoextra)
UCI_data <- read.table(file = 'wdbc.data', header = TRUE)
names(UCI_data)
names <- c('id', 'diagnosis', 'radius_mean', 
           'texture_mean', 'perimeter_mean', 'area_mean', 
           'smoothness_mean', 'compactness_mean', 
           'concavity_mean','concave_points_mean', 
           'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 
           'area_se', 'smoothness_se', 'compactness_se', 
           'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 
           'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 
           'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 
           'symmetry_worst', 'fractal_dimension_worst')
bc_data <- read.table(file = 'wdbc.data', sep = ',', col.names = names)
str(bc_data)
head(bc_data)
dim(bc_data)

# Identify and Remove Duplicate Data
sum(duplicated(bc_data))

# extracting positions of NA values
sum(is.na(bc_data))


# Convert bc_data into factor variable
bc_data$diagnosis <- as.factor(bc_data$diagnosis)

# Drop id column
bc_data_clean <- bc_data[-1]

dim(bc_data_clean)

# The frequency distribution of benign and malignant groups
round(prop.table(table(bc_data_clean$diagnosis)), 2)
ggplot(bc_data_clean, aes(x = diagnosis, fill = diagnosis)) +
  geom_bar(show.legend = FALSE) +
  geom_label(aes(label = paste(..count.., " (", round(prop.table(..count..) * 100, 2), "%)")), 
             stat = "count", 
             position = position_stack(vjust = 0.75),
             show.legend = FALSE) +
  labs(title = "Tumor type distribution", x = "diagnosis", y = "Frequency")

# Quick look at variables distribution
boxplot(bc_data_clean)
ggpairs(bc_data_clean[1:10], aes(colour = bc_data_clean$diagnosis, alpha = 0.6))
ggpairs(bc_data_clean[11:20], aes(colour = bc_data_clean$diagnosis, alpha = 0.6))
ggpairs(bc_data_clean[21:31], aes(colour = bc_data_clean$diagnosis, alpha = 0.6))

#checking the between and within set associations/correlation
df_corr <- cor(bc_data_clean[,2:31])
corrplot(df_corr, order = "hclust", tl.cex = 0.7, addrect = 8)


# remove highly correlated predictors based on whose correlation is above 0.9.

# highlyCor:highly correlated predictors
(highlyCor <- colnames(bc_data_clean)[findCorrelation(df_corr, cutoff = 0.9, verbose = TRUE)])
bc_data_cor <- bc_data[, which(!colnames(bc_data_clean) %in% highlyCor)]
bc_data_cor <- cbind(diagnosis = bc_data_clean$diagnosis, bc_data_cor)
dim(bc_data_cor)
str(bc_data_cor)

# Principal Components Analysis 
cancer.pca <- prcomp(bc_data_cor[,-c(1)], center=TRUE, scale=TRUE)
summary(cancer.pca)

# plot Principal components weight
plot(cancer.pca, type="l", main='')
grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()

# Calculate the proportion of variance explained
loadings<-cancer.pca$rotation
(loadings<-round(loadings, digits=2))
(pervar<-((cancer.pca$sdev^2/sum(cancer.pca$sdev^2))*100))
(pervar<-round(pervar,digits=1))
fviz_eig(cancer.pca, addlabels = TRUE, ylim = c(0, 50))

# Plots PC 1 and PC2
pca_df <- as.data.frame(cancer.pca$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=bc_data_clean$diagnosis)) + geom_point(alpha=0.5)

df_pcs <- cbind(as_tibble(bc_data_clean$diagnosis), as_tibble(cancer.pca$x))
GGally::ggpairs(df_pcs, columns = 2:3, ggplot2::aes(color = value))

autoplot(cancer.pca, data = bc_data_clean,  colour = 'diagnosis',
         loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue")



# Applying machine learning models
# Split data set in train 70% and test 30%
# Train the algorithm 
# make predictions
# evaluate the predictions against the expected results.

set.seed(1234)
train_indx <- createDataPartition(bc_data_cor$diagnosis, p = 0.7, list = FALSE)

train_set <- bc_data_cor[train_indx,]
test_set <- bc_data_cor[-train_indx,]

nrow(train_set)
nrow(test_set)


fitControl <- trainControl(method="repeatedcv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           repeats = 10)

model_lr <- train(diagnosis ~.,
                  data = train_set,
                  method = "glm", 
                  metric = "ROC",
                  preProcess = c("scale", "center"), 
                  trControl = fitControl)

pred_lr <- predict(model_lr, test_set)
cm_lr <- confusionMatrix(pred_lr, test_set$diagnosis, positive = "M")
cm_lr

model_rf <- train(diagnosis~.,
                  data = train_set,
                  method="rf",
                  metric="ROC",
                  #tuneLength=10,
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)
# plot feature importance
varImp(model_rf)
plot(varImp(model_rf), top = 10, main = "Random forest")

# Feed test data to the model
pred_rf <- predict(model_rf, test_set)
cm_rf <- confusionMatrix(pred_rf, test_set$diagnosis, positive = "M")
cm_rf

# Random Forest with PCA
model_pca_rf <- train(diagnosis~.,
                      data = train_set,
                      method="ranger",
                      metric="ROC",
                      #tuneLength=10,
                      preProcess = c('center', 'scale', 'pca'),
                      trControl=fitControl)

pred_pca_rf <- predict(model_pca_rf, test_set)
cm_pca_rf <- confusionMatrix(pred_pca_rf, test_set$diagnosis, positive = "M")
cm_pca_rf

#KNN
model_knn <- train(diagnosis~.,
                   data = train_set,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)

pred_knn <- predict(model_knn, test_set)
cm_knn <- confusionMatrix(pred_knn, test_set$diagnosis, positive = "M")
cm_knn 

#Neural Networks (NNET)

model_nnet <- train(diagnosis~.,
                    data = train_set,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    tuneLength=10,
                    trControl=fitControl)

pred_nnet <- predict(model_nnet, test_set)
cm_nnet <- confusionMatrix(pred_nnet, test_set$diagnosis, positive = "M")
cm_nnet

#Neural Networks (NNET) with PCA
model_pca_nnet <- train(diagnosis~.,
                        data = train_set,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)

pred_pca_nnet <- predict(model_pca_nnet, test_set)
cm_pca_nnet <- confusionMatrix(pred_pca_nnet, test_set$diagnosis, positive = "M")
cm_pca_nnet

# SVM with radial kernel
model_svm <- train(diagnosis~.,
                   data = train_set,
                   method="svmRadial",
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)

pred_svm <- predict(model_svm, test_set)
cm_svm <- confusionMatrix(pred_svm, test_set$diagnosis, positive = "M")
cm_svm

# Naive Bayes
model_nb <- train(diagnosis~.,
                  data = train_set,
                  method="nb",
                  metric="ROC",
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  trControl=fitControl)

pred_nb <- predict(model_nb, test_set)
cm_nb <- confusionMatrix(pred_nb, test_set$diagnosis, positive = "M")
cm_nb

# Models evaluation
model_list <- list(LR=model_lr, RF=model_rf, PCA_RF=model_pca_rf, 
                   NNET=model_nnet, PCA_NNET=model_pca_nnet,  
                   KNN = model_knn, SVM=model_svm, NB=model_nb)
resamples <- resamples(model_list)
bwplot(resamples, metric = "ROC")

cm_list <- list(LR = cm_lr, RF=cm_rf, PCA_RF=cm_pca_rf, 
                NNET=cm_nnet, PCA_NNET=cm_pca_nnet,  
                KNN = cm_knn, SVM=cm_svm, NB=cm_nb)

results <- sapply(cm_list, function(x) x$byClass)
round(results, 3)

results_max <- apply(results, 1, which.max)
output_report <- data.frame(metric=names(results_max), 
                            best_model=colnames(results)[results_max],
                            value=mapply(function(x,y) {results[x,y]}, 
                                         names(results_max), 
                                         results_max))
rownames(output_report) <- NULL
output_report

