###############################
## PROBIT REGRESSION MODEL ##
#### PREDICTING CREDIT POLICY ###
###############################

# Load libraries
library(caret)
library(rsample)
library(pROC)
library(tidymodels)

# Load and prepare data
df <- read.csv("/Users/yoyo/Desktop/Levkoff Final Project/loan_data.csv")
df$purpose <- factor(df$purpose)
df$credit.policy <- factor(df$credit.policy)

# Stratified 70/15/15 split
set.seed(123)
split1 <- initial_split(df, prop = 0.7, strata = "credit.policy")
train <- training(split1)
temp <- testing(split1)
split2 <- initial_split(temp, prop = 0.5, strata = "credit.policy")
validation <- training(split2)
test <- testing(split2)

# Verify class distribution
cat("Training set class distribution:\n")
print(table(train$credit.policy))
cat("\nValidation set class distribution:\n")
print(table(validation$credit.policy))
cat("\nTest set class distribution:\n")
print(table(test$credit.policy))

#############################################
### IN-SAMPLE (TRAINING) EVALUATION FIRST ###
#############################################

# Train probit model
probit_model <- glm(credit.policy ~ .,
                    data = train,
                    family = binomial(link = "probit"))

# In-sample predictions
train_probs <- predict(probit_model, train, type = "response")
train_class <- ifelse(train_probs >= 0.5, 1, 0)

# In-sample confusion matrix
cat("\n=== IN-SAMPLE (TRAINING) PERFORMANCE ===\n")
train_confusion <- confusionMatrix(factor(train_class), train$credit.policy, positive = "1")
print(train_confusion)

# Calculate in-sample error
E_IN <- 1 - train_confusion$overall["Accuracy"]
cat("\nIn-sample error (E_IN):", round(E_IN, 4), "\n")

# In-sample ROC/AUC
train_roc <- roc(train$credit.policy, train_probs)
plot(train_roc, main = "In-Sample ROC Curve")
cat("In-sample AUC:", auc(train_roc), "\n")

#############################################
### COEFFICIENT TABLE FOR VARIABLE EVALUATION ###
#############################################

# Create comprehensive coefficient table
coef_table <- data.frame(
  Variable = names(coef(probit_model)),
  Estimate = round(coef(probit_model), 4),
  Std.Error = round(summary(probit_model)$coefficients[, "Std. Error"], 4),
  z.value = round(summary(probit_model)$coefficients[, "z value"], 2),
  p.value = round(summary(probit_model)$coefficients[, "Pr(>|z|)"], 4),
  OR_Approx = round(exp(coef(probit_model)), 4)  # Approximate odds ratio
)

# Sort by absolute effect size (largest first)
coef_table <- coef_table[order(-abs(coef_table$Estimate)), ]

# Print formatted table
cat("\n=== PROBIT REGRESSION COEFFICIENTS ===\n")
print(coef_table, row.names = FALSE)

# Interpretation guide
cat("\nKEY TO INTERPRETATION:
- Estimate: Probit coefficient (z-score change)
- OR_Approx: Approximate odds ratio (e^Estimate)
- p.value < 0.05 = Significant predictor\n")

#############################################
### VALIDATION SET EVALUATION ###
#############################################

# Validation set predictions
val_probs <- predict(probit_model, validation, type = "response")
val_class <- ifelse(val_probs >= 0.5, 1, 0)

cat("\n=== VALIDATION SET PERFORMANCE ===\n")
val_confusion <- confusionMatrix(factor(val_class), validation$credit.policy, positive = "1")
print(val_confusion)

# Calculate validation error
E_VAL <- 1 - val_confusion$overall["Accuracy"]
cat("\nValidation error (E_VAL):", round(E_VAL, 4), "\n")

# Validation ROC/AUC
val_roc <- roc(validation$credit.policy, val_probs)
plot(val_roc, main = "Validation ROC Curve")
cat("Validation AUC:", auc(val_roc), "\n")

#############################################
### TEST SET EVALUATION ###
#############################################

# Test set predictions
test_probs <- predict(probit_model, test, type = "response")
test_class <- ifelse(test_probs >= 0.5, 1, 0)

cat("\n=== TEST SET PERFORMANCE ===\n")
test_confusion <- confusionMatrix(factor(test_class), test$credit.policy, positive = "1")
print(test_confusion)

# Calculate test error
E_OUT <- 1 - test_confusion$overall["Accuracy"]
cat("\nTest error (E_OUT):", round(E_OUT, 4), "\n")

# Test ROC/AUC
test_roc <- roc(test$credit.policy, test_probs)
plot(test_roc, 
     main = "ROC Curve (Test Set)", 
     print.auc = TRUE, 
     auc.polygon = TRUE, 
     grid = TRUE,
     legacy.axes = TRUE)  # Shows False Positive Rate on x-axis

# Store AUC value for later use if needed
test_auc <- auc(test_roc)

#############################################
### COMPREHENSIVE PERFORMANCE SUMMARY ###
#############################################

# Create performance comparison table
performance_summary <- data.frame(
  Dataset = c("Training", "Validation", "Test"),
  Accuracy = c(train_confusion$overall["Accuracy"],
               val_confusion$overall["Accuracy"],
               test_confusion$overall["Accuracy"]),
  Sensitivity = c(train_confusion$byClass["Sensitivity"],
                  val_confusion$byClass["Sensitivity"],
                  test_confusion$byClass["Sensitivity"]),
  Specificity = c(train_confusion$byClass["Specificity"],
                  val_confusion$byClass["Specificity"],
                  test_confusion$byClass["Specificity"]),
  AUC = c(auc(train_roc), auc(val_roc), auc(test_roc))
)

cat("\n=== MODEL PERFORMANCE ACROSS DATASETS ===\n")
print(performance_summary, row.names = FALSE)