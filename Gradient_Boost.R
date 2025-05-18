##################################
## GRADIENT BOOSTING MODEL      ##
##################################

#LOADING THE LIBRARIES
library(tidymodels)
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()

#IMPORTING THE DATA
df <- read.csv("/Users/yoyo/Desktop/Levkoff Final Project/loan_data.csv")
df$credit.policy <- as.factor(df$credit.policy) #CONVERT OUTPUT TO FACTOR

##PARTITIONING THE DATA##
set.seed(123)
split <- initial_split(df, prop = .7, strata = credit.policy)
train <- training(split)
test <- testing(split)

#MODEL DESCRIPTION:
fmla <- credit.policy ~ .

###################################
#SPECIFYING GRADIENT BOOSTED MODEL#
###################################

#SPECIFY AND FIT IN ONE STEP:
boosted_forest <- boost_tree(min_n = NULL, #minimum number of observations for split
                             tree_depth = NULL, #max tree depth
                             trees = 100, #number of trees
                             mtry = NULL, #number of predictors selected at each split 
                             sample_size = NULL, #amount of data exposed to fitting
                             learn_rate = NULL, #learning rate for gradient descent
                             loss_reduction = NULL, #min loss reduction for further split
                             stop_iter = NULL)  %>% #maximum iteration for convergence
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  fit(fmla, train)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_xb_in <- predict(boosted_forest, new_data = train, type = "class") %>%
  bind_cols(train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_xb_in <- table(pred_class_xb_in$.pred_class, pred_class_xb_in$credit.policy)
confusionMatrix(confusion_xb_in) #FROM CARET PACKAGE

#STORE E_IN
E_IN_XB <- 1 - sum(diag(as.matrix(confusion_xb_in)))/nrow(train)

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_xb_out <- predict(boosted_forest, new_data = test, type = "class") %>%
  bind_cols(test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_xb_out <- table(pred_class_xb_out$.pred_class, pred_class_xb_out$credit.policy)
confusionMatrix(confusion_xb_out) #FROM CARET PACKAGE

#STORE E_OUT
E_OUT_XB <- 1 - sum(diag(as.matrix(confusion_xb_out)))/nrow(test)

#PRINT RESULTS
cat("Gradient Boosted Model Performance:\n")
cat("In-sample error (E_IN):", E_IN_XB, "\n")
cat("Out-of-sample error (E_OUT):", E_OUT_XB, "\n")

