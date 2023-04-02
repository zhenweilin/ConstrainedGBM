## drybean dataset
rm(list = ls())
library(npcs)
library(caTools)

## define a function to train_test_split
train_test_split <- function(X, y, ratio = 0.2,seed = 0){
  set.seed(seed)
  split <- sample.split(y, SplitRatio = ratio)
  X_train <- subset(X, split == FALSE)
  y_train <- subset(y, split == FALSE)
  X_test <- subset(X, split == TRUE)
  y_test <- subset(y, split == TRUE)
  res <- list(X_train = X_train, y_train = y_train,X_test = X_test, y_test = y_test)
  return(res)
}

load("../../data/DryBeanDataset/X_dryBean.gzip")
load("../../data/DryBeanDataset/y_dryBean.gzip")
X <- X_dryBean
y <- y_dryBean


col_names <- c('dataset', 'model', 'constraint_violation(train)', 'accuracy(train)', 'constraint_violation(test)', 'accuracy(test)')

for (random_state in c(1:2) ){
  file_name <- '../save_csv/drybean_np_r.csv'
  
  if (!file.exists(file_name)){
    empty_df <- data.frame(matrix(ncol = length(col_names), nrow = 0))
    colnames(empty_df) <- col_names
    write.csv(empty_df, file_name, row.names = FALSE)
  }
  
  dataset <- train_test_split(X, y, ratio = 0.2, seed = random_state)
  X_train <- dataset$X_train
  y_train <- dataset$y_train
  X_test <- dataset$X_test
  y_test <- dataset$y_test
  
  w <- c(1, 1, 1, 1, 1, 1, 1)
  wmatrix = matrix(w)
  alpha <- c(NA, 0.05, 0.05, 0.04, 0.05, NA, NA)
  alpha_compare <- c(1, 0.05, 0.05, 0.04, 0.05, 1, 1)
  y_train <- factor(y_train)
  set.seed(random_state)
  fit.npmc.CX <- try(npcs(X_train, y_train, algorithm = "CX", classifier = "randomforest", w = w, alpha = alpha, ntree = floor(rnorm(1, mean = 500, sd = 50))))
  fit.npmc.ER <- try(npcs(X_train, y_train, algorithm = "ER", classifier = "randomforest", w = w, alpha = alpha,
                          refit = TRUE, ntree = floor(rnorm(1, mean = 500, sd = 50))))
  
  npcs_cx_row <- c('drybean', 'npcs(cx)')
  ########## train error cx #############
  train_y_pred <- predict(fit.npmc.CX, X_train)
  err_tr <- error_rate(train_y_pred, y_train)
  err_train_cx <- matrix(err_tr, ncol = length(alpha))
  err_obj <- err_tr%*%wmatrix
  constraint_violation_train_cx <- sum(pmax(err_train_cx - alpha_compare, 0))
  acc_train_cx <- 1 - err_obj
  npcs_cx_row <- c(npcs_cx_row, constraint_violation_train_cx, acc_train_cx)
  
  ########### test error cx #############
  ytest <- factor(y_test)
  y_pred <- predict(fit.npmc.CX, X_test)
  err_te <- error_rate(y_pred, y_test)
  err_test_cx <- matrix(err_te, ncol = length(alpha))
  err_obj <- err_test_cx%*%wmatrix
  constraint_violation_test_cx <- sum(pmax(err_test_cx - alpha_compare, 0))
  acc_test_cx <- 1- err_obj
  npcs_cx_row <- c(npcs_cx_row, constraint_violation_test_cx, acc_test_cx)
  
  npcs_er_row <- c('drybean', 'npcs(er)')
  ############ train error er ############
  train_y_pred <- predict(fit.npmc.ER, X_train)
  err_tr <- error_rate(train_y_pred, y_train)
  err_train_er <- matrix(err_tr, ncol = length(alpha))
  err_obj <- err_train_er %*% wmatrix
  constraint_violation_train_er <- sum(pmax(err_train_er - alpha_compare, 0))
  acc_train_er <- 1 - err_obj
  npcs_er_row <- c(npcs_er_row, constraint_violation_train_er, acc_train_er)
  
  ############# test error er #############
  ytest <- factor(y_test)
  y_pred <- predict(fit.npmc.ER, X_test)
  err_te <- error_rate(y_pred, y_test)
  err_test_er <- matrix(err_te, ncol = length(alpha))
  err_obj <- err_test_er%*%wmatrix
  constraint_violation_test_er <- sum(pmax(err_test_er - alpha_compare, 0))
  acc_test_er <- 1 - err_obj
  npcs_er_row <- c(npcs_er_row, constraint_violation_test_er, acc_test_er)
  
  write.table(matrix(npcs_cx_row, ncol = length(npcs_cx_row)), file_name, append = TRUE, row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(matrix(npcs_er_row, ncol = length(npcs_er_row)), file_name, append = TRUE, row.names = FALSE, col.names = FALSE, sep = ",")
}
