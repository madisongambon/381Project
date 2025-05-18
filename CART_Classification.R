######################################
## CLASSIFICATION TREE FOR CREDIT POLICY ##
######################################

# Load only essential packages
library(rpart)        # For decision trees
library(rpart.plot)   # For tree visualization
library(caret)        # For confusionMatrix()
library(pROC)         # For ROC analysis
library(tidymodels)   # For data splitting

# Load data
df <- read.csv("/Users/yoyo/Desktop/Levkoff Final Project/loan_data.csv")
df$credit.policy <- as.factor(df$credit.policy)  # Convert target to factor

# Convert categorical predictors to factors
df$purpose <- as.factor(df$purpose)  # Example - add others as needed

# Create three partitions (train, validation, test) with stratification
set.seed(123)
# First split: 70% train, 30% validation+test
split1 <- initial_split(df, prop = 0.7, strata = "credit.policy")
train <- training(split1)
val_test <- testing(split1)

# Second split: 50% validation, 50% test of remaining 30%
split2 <- initial_split(val_test, prop = 0.5, strata = "credit.policy")
validation <- training(split2)
test <- testing(split2)

# Verify class distribution
cat("Training set class distribution:\n")
print(table(train$credit.policy))
cat("\nValidation set class distribution:\n")
print(table(validation$credit.policy))
cat("\nTest set class distribution:\n")
print(table(test$credit.policy))

# Build initial classification tree
tree_model <- rpart(credit.policy ~ .,
                    data = train,
                    method = "class",
                    control = rpart.control(
                      minsplit = 20,     # Min observations to split
                      maxdepth = 5,      # Max tree depth
                      cp = 0.01          # Complexity parameter
                    ))

# Visualize the initial tree
rpart.plot(tree_model, 
           type = 4, 
           extra = 104,
           box.palette = "Blues",
           shadow.col = "gray",
           main = "Initial Classification Tree")

# Prune the tree using validation set
printcp(tree_model)  # View complexity table

# Find optimal cp using validation set
val_pred_prob <- predict(tree_model, validation, type = "prob")[,2]
val_roc <- roc(validation$credit.policy, val_pred_prob)
best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"]

pruned_tree <- prune(tree_model, cp = best_cp)

# Visualize pruned tree
rpart.plot(pruned_tree,
           type = 4,
           extra = 104,
           box.palette = "Greens",
           shadow.col = "gray",
           main = "Pruned Classification Tree")

# Make predictions on validation set for early evaluation
validation$pred_class <- predict(pruned_tree, validation, type = "class")
validation$pred_prob <- predict(pruned_tree, validation, type = "prob")[,2]

# Validation set performance
cat("\nValidation Set Performance:\n")
val_confusion <- confusionMatrix(validation$pred_class, 
                                 validation$credit.policy, 
                                 positive = "1")
print(val_confusion)

# Validation ROC Curve
val_roc <- roc(validation$credit.policy, validation$pred_prob)
plot(val_roc, main = "Validation ROC Curve")
cat("\nValidation AUC:", auc(val_roc), "\n")

# Final evaluation on test set
test$pred_class <- predict(pruned_tree, test, type = "class")
test$pred_prob <- predict(pruned_tree, test, type = "prob")[,2]

# Test set performance
cat("\nTest Set Performance:\n")
test_confusion <- confusionMatrix(test$pred_class, 
                                  test$credit.policy, 
                                  positive = "1")
print(test_confusion)

# Test ROC Curve
test_roc <- roc(test$credit.policy, test$pred_prob)
plot(test_roc, main = "Test ROC Curve")
cat("\nTest AUC:", auc(test_roc), "\n")

# Variable Importance
importance <- pruned_tree$variable.importance
barplot(sort(importance, decreasing = TRUE), 
        las = 2, 
        main = "Variable Importance",
        col = "lightblue")

