###############################
## LOGISTIC REGRESSION MODEL ##
#### PREDICTING CREDIT POLICY ###
###############################

# Load data and libraries
df <- read.csv('/Users/yoyo/Desktop/Levkoff Final Project/loan_data.csv')
library(caret)
library(rsample)
library(pROC)
library(dplyr)

# Convert categorical variables
df$purpose <- factor(df$purpose)
df$credit.policy <- factor(df$credit.policy)

# Stratified split (70% train, 15% validation, 15% test)
set.seed(456)
split1 <- initial_split(df, prop = 0.7, strata = "credit.policy")
train <- training(split1)
temp <- testing(split1)
split2 <- initial_split(temp, prop = 0.5, strata = "credit.policy")
validation <- training(split2)
test <- testing(split2)

# Train model
model <- glm(credit.policy ~ ., data = train, family = "binomial")

#############################################
### IN-SAMPLE EVALUATION (Training Set) ###
#############################################

# Training set predictions
train_probs <- predict(model, train, type = "response")
train_pred <- ifelse(train_probs >= 0.5, 1, 0)

# Training performance
train_cm <- confusionMatrix(factor(train_pred), train$credit.policy, positive = "1")
train_accuracy <- train_cm$overall["Accuracy"]
train_auc <- auc(roc(train$credit.policy, train_probs))

cat("\n=== IN-SAMPLE PERFORMANCE (TRAINING SET) ===\n")
cat("Accuracy:", round(train_accuracy, 4), "\n")
cat("AUC:", round(train_auc, 4), "\n\n")
print(train_cm$table)

#############################################
### VALIDATION SET EVALUATION ###
#############################################

# Validation set predictions
val_probs <- predict(model, validation, type = "response")
val_pred <- ifelse(val_probs >= 0.5, 1, 0)

# Validation performance
val_cm <- confusionMatrix(factor(val_pred), validation$credit.policy, positive = "1")
val_accuracy <- val_cm$overall["Accuracy"]
val_auc <- auc(roc(validation$credit.policy, val_probs))

cat("\n=== VALIDATION SET PERFORMANCE ===\n")
cat("Accuracy:", round(val_accuracy, 4), "\n")
cat("AUC:", round(val_auc, 4), "\n\n")
print(val_cm$table)

#############################################
### OUT-OF-SAMPLE EVALUATION (Test Set) ###
#############################################

# Test set predictions
test_probs <- predict(model, test, type = "response")
test_pred <- ifelse(test_probs >= 0.5, 1, 0)

# Test performance
test_cm <- confusionMatrix(factor(test_pred), test$credit.policy, positive = "1")
test_accuracy <- test_cm$overall["Accuracy"]
test_auc <- auc(roc(test$credit.policy, test_probs))

cat("\n=== OUT-OF-SAMPLE PERFORMANCE (TEST SET) ===\n")
cat("Accuracy:", round(test_accuracy, 4), "\n")
cat("AUC:", round(test_auc, 4), "\n\n")
print(test_cm$table)

# ROC Curve
plot(roc(test$credit.policy, test_probs), 
     main = "Test Set ROC Curve",
     col = "blue",
     print.auc = TRUE)

#############################################
### PERFORMANCE SUMMARY ###
#############################################

performance_summary <- data.frame(
  Dataset = c("Training (In-Sample)", "Validation", "Test (Out-of-Sample)"),
  Accuracy = c(train_accuracy, val_accuracy, test_accuracy),
  AUC = c(train_auc, val_auc, test_auc)
)

cat("\n=== MODEL PERFORMANCE SUMMARY ===\n")
print(performance_summary, row.names = FALSE)

#############################################
### VARIABLE IMPORTANCE ###
#############################################

# Get coefficients and sort by impact
coef_df <- data.frame(
  Variable = names(coef(model)),
  Coefficient = coef(model),
  Odds_Ratio = exp(coef(model)),
  p_value = summary(model)$coefficients[,4]
) %>% 
  arrange(desc(abs(Odds_Ratio - 1)))  # Sort by impact magnitude

cat("\n=== VARIABLE IMPACT SORTED BY ODDS RATIO ===\n")
print(coef_df, row.names = FALSE)