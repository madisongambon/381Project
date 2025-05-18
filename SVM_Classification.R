###############################################
## SVM CLASSIFIER FOR CREDIT POLICY PREDICTION ##
###############################################

library(rsample)  # For initial_split()
library(e1071)    # For SVM
library(caret)    # For confusionMatrix()
library(pROC)     # For ROC curve

# Load your data
df <- read.csv("/Users/yoyo/Desktop/Levkoff Final Project/loan_data.csv")
df$credit.policy <- as.factor(df$credit.policy)  # Convert target to factor

# View data structure
str(df)
summary(df)

####################################################
# IMPORTANT: Convert all categorical predictors to numeric
# Example for 'purpose' column:
df$purpose_numeric <- as.numeric(factor(df$purpose))
# Drop original categorical columns after conversion
df <- df[, !(names(df) %in% c("purpose"))]  # Add other categorical columns
####################################################

# Stratified train-test split
set.seed(123)
split <- initial_split(df, 0.7, strata = credit.policy)
training <- training(split)
testing <- testing(split)

# Verify class balance
prop.table(table(training$credit.policy))
prop.table(table(testing$credit.policy))

###### KERNEL SELECTION ######
kern_type <- "radial"  # Options: "linear", "polynomial", "radial", "sigmoid"

# Initial untuned SVM
svm_untuned <- svm(credit.policy ~ .,
                   data = training,
                   type = "C-classification",
                   kernel = kern_type,
                   cost = 1,
                   gamma = 1/(ncol(training)-1),
                   scale = TRUE)  # Important to scale for SVM

# Untuned performance
train_pred_untuned <- predict(svm_untuned, training)
test_pred_untuned <- predict(svm_untuned, testing)

cat("Untuned Model Performance:\n")
confusionMatrix(train_pred_untuned, training$credit.policy, positive = "1")
confusionMatrix(test_pred_untuned, testing$credit.policy, positive = "1")

###### HYPERPARAMETER TUNING ######
set.seed(123)
tune_control <- tune.control(cross = 5)  # 5-fold CV

svm_tune <- tune.svm(credit.policy ~ .,
                     data = training,
                     type = "C-classification",
                     kernel = kern_type,
                     tunecontrol = tune_control,
                     cost = c(0.1, 1, 10),
                     gamma = c(0.01, 0.1))
                     
print(svm_tune$best.parameters)

# Retuned model
svm_tuned <- svm(credit.policy ~ .,
                 data = training,
                 type = "C-classification",
                 kernel = kern_type,
                 cost = svm_tune$best.parameters$cost,
                 gamma = svm_tune$best.parameters$gamma,
                 scale = TRUE,
                 probability = TRUE)

# Tuned performance
train_pred_tuned <- predict(svm_tuned, training)
test_pred_tuned <- predict(svm_tuned, testing)

#############################
### FINAL MODEL EVALUATION ###
#############################

# IN-SAMPLE (Training set)
cat("\n=== IN-SAMPLE (TRAINING) PERFORMANCE ===\n")
confusionMatrix(train_pred_tuned, training$credit.policy, positive = "1")

# OUT-OF-SAMPLE (Test set)
cat("\n=== OUT-OF-SAMPLE (TESTING) PERFORMANCE ===\n")
confusionMatrix(test_pred_tuned, testing$credit.policy, positive = "1")

# Compute and print error rates
train_error <- 1 - mean(train_pred_tuned == training$credit.policy)
test_error <- 1 - mean(test_pred_tuned == testing$credit.policy)

cat(sprintf("\nTrain Error (in-sample): %.4f", train_error))
cat(sprintf("\nTest Error (out-of-sample): %.4f", test_error))

###### ROC/AUC ANALYSIS ######

# Get probability predictions (requires probability = TRUE in model)
svm_probs <- attr(predict(svm_tuned, testing, probability = TRUE), "probabilities")[, "1"]
roc_obj <- roc(testing$credit.policy, svm_probs)

plot(roc_obj, main = paste("SVM", kern_type, "Kernel ROC Curve"))
auc(roc_obj)

###### PERFORMANCE COMPARISON ######
results <- data.frame(
  Model = c("Untuned", "Tuned"),
  In_Sample_Accuracy = round(c(
    mean(train_pred_untuned == training$credit.policy),
    mean(train_pred_tuned == training$credit.policy)
  ), 4),
  Out_of_Sample_Accuracy = round(c(
    mean(test_pred_untuned == testing$credit.policy),
    mean(test_pred_tuned == testing$credit.policy)
  ), 4),
  In_Sample_Error = round(c(
    1 - mean(train_pred_untuned == training$credit.policy),
    1 - mean(train_pred_tuned == training$credit.policy)
  ), 4),
  Out_of_Sample_Error = round(c(
    1 - mean(test_pred_untuned == testing$credit.policy),
    1 - mean(test_pred_tuned == testing$credit.policy)
  ), 4)
)

cat("\n=== FINAL PERFORMANCE COMPARISON ===\n")
print(results)


#---
# Fit SVM model

# Print summary
summary(svm_tuned)

# If linear, extract weights (only for linear kernel)
w <- t(svm_model$coefs) %*% svm_model$SV
print("Feature weights (w):")
print(w)
