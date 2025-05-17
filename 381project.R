df<- read.csv("/Users/madisongambon/Desktop/loan_data.csv")
View(df)
library(tidyverse)    # data munging + ggplot2
library(caret)        # train/validation split, resampling
library(glmnet)       # regularization
library(mgcv)         # GAM / splines

library(dplyr)
library(corrplot)

# Select numeric columns
numeric_df <- df %>% select(where(is.numeric))

# Compute correlation matrix
cor_matrix <- cor(numeric_df, use = "complete.obs")

# Make plot with bigger margins and clearer spacing
corrplot(cor_matrix,
         method = "color",
         type = "lower",
         order = "hclust",
         tl.col = "black",
         tl.cex = 1.2,               # bigger label text
         tl.srt = 45,                # angled labels
         addCoef.col = "black",     # correlation numbers
         number.cex = 0.8,           # slightly smaller numbers
         col = colorRampPalette(c("steelblue", "white", "darkred"))(200),
         mar = c(1, 1, 1, 1),
         cl.cex = 1.2,               # bigger color legend text
         title = "Loan Data Correlation Matrix",
         addgrid.col = "grey90")


# Set seed for reproducibility (very important for consistency across group members)
set.seed(123)

# Remove any rows with missing values (optional but common in preprocessing)
df <- na.omit(df)

# Shuffle the rows randomly
df <- df[sample(nrow(df)), ]

# Total number of observations
n <- nrow(df)

# Create row indices for each split
train_index <- 1:floor(0.6 * n)
valid_index <- (floor(0.6 * n) + 1):floor(0.8 * n)
test_index  <- (floor(0.8 * n) + 1):n

# Split the dataset
train_set <- df[train_index, ]
valid_set <- df[valid_index, ]
test_set  <- df[test_index, ]

# Save each partition to a CSV file for sharing and reproducibility
write.csv(train_set, "train_set.csv", row.names = FALSE)
write.csv(valid_set, "valid_set.csv", row.names = FALSE)
write.csv(test_set,  "test_set.csv", row.names = FALSE)

# Confirm the sizes
cat("Partitioning Complete\n")
cat("Training set:", nrow(train_set), "rows\n")
cat("Validation set:", nrow(valid_set), "rows\n")
cat("Testing set:", nrow(test_set), "rows\n")

# Recombine the character vector (loan_data)
loan_text <- paste(df, collapse = "\n")

# Read just the first row to extract the real column names
header_line <- read.csv(text = loan_text, header = FALSE, nrows = 1, stringsAsFactors = FALSE)
col_names <- unlist(header_line)

# Now read the full dataset, skipping the first row, and applying the column names manually
df <- read.csv(text = loan_text, header = FALSE, skip = 1, stringsAsFactors = FALSE)
colnames(df) <- col_names

##BIVARIATE REGRESSION MODEL BUILDING, LINEAR REGRESSION### part a #
lm1 <- lm(train_set$int.rate ~ train_set$fico, data = train_set)
summary(lm1)

pred_train1 <- predict(lm1, train_set)
rmse_train1 <- RMSE(pred_train1, train_set$int.rate)
rmse_train1

library(ggplot2)

# 1. Compute fitted values, residuals, and their absolute magnitude
train_set$fitted    <- predict(lm1, train_set)
train_set$resid     <- train_set$int.rate - train_set$fitted
train_set$abs_resid <- abs(train_set$resid)

# 2. Plot Residuals vs Fitted, coloring by |residual|
ggplot(train_set, aes(x = fitted, y = resid, color = abs_resid)) +
  geom_point(alpha = 0.8, size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_gradient(
    low  = "lightblue",
    high = "darkblue",
    name = "|Residual|"
  ) +
  labs(
    title = "Residuals vs Predicted Values\nColor ~ Distance from Zero",
    x     = "Predicted Values",
    y     = "Residuals (Actual − Predicted)"
  ) +
  theme_minimal()

coefs <- coef(lm1)
ggplot(train_set, aes(x = fico, y = int.rate)) +
  geom_point(alpha = 0.6) +
  geom_abline(
    intercept = coefs[1],
    slope     = coefs[2],
    color     = "blue",
    linewidth  = 1.2
  ) +
  theme_minimal() +
  labs(
    title = "Actual Data with Linear Model Line",
    x     = "FICO Score",
    y     = "Interest Rate"
  )

#4B CODE#

model_linear <- lm(int.rate ~ fico, data = train_set)
summary(model_linear)
# Polynomial model (adds fico^2)
train_set$fico2 <- train_set$fico^2
valid_set$fico2 <- valid_set$fico^2
model_poly <- lm(int.rate ~ fico + fico2, data = train_set)
summary(model_poly)
library(ggplot2)


# Predictions
pred_linear <- predict(model_linear, newdata = valid_set)
pred_poly <- predict(model_poly, newdata = valid_set)

# Error metrics
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2 <- function(actual, pred) cor(actual, pred)^2

# Results
cat("Linear Model RMSE:", rmse(valid_set$int.rate, pred_linear), "\n")
cat("Linear Model R²:", r2(valid_set$int.rate, pred_linear), "\n\n")

cat("Polynomial Model RMSE:", rmse(valid_set$int.rate, pred_poly), "\n")
cat("Polynomial Model R²:", r2(valid_set$int.rate, pred_poly), "\n")

# Create a data frame for plotting with actual FICO scores and predictions
plot_data <- data.frame(
  fico = valid_set$fico,
  int_rate_actual = valid_set$int.rate,
  int_rate_poly = pred_poly
)

# Sort data by fico score for a smooth curve
plot_data <- plot_data[order(plot_data$fico), ]

# Plot actual vs predicted (polynomial)
ggplot(plot_data, aes(x = fico)) +
  geom_point(aes(y = int_rate_actual), color = "black", alpha = 0.5, size = 1.8) +
  geom_line(aes(y = int_rate_poly), color = "#D55E00", size = 1.2) +
  labs(
    title = "Polynomial Model vs Actual Data",
    x = "FICO Score",
    y = "Interest Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
  )


#4C: Regularization with Tuning

# Load library
library(glmnet)

# Make sure predictor terms are ready
train_set$fico2 <- train_set$fico^2
valid_set$fico2 <- valid_set$fico^2

# Prepare input matrices (X) and output vector (y)
X_train <- as.matrix(train_set[, c("fico", "fico2")])
y_train <- train_set$int.rate

X_valid <- as.matrix(valid_set[, c("fico", "fico2")])
y_valid <- valid_set$int.rate

# Run cross-validated Ridge regression (alpha = 0)
set.seed(123)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)

# Best lambda value
best_lambda <- cv_ridge$lambda.min

# Predict on validation set using best lambda
pred_ridge <- predict(cv_ridge, s = best_lambda, newx = X_valid)

# Evaluation metrics
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2 <- function(actual, pred) cor(actual, pred)^2

cat("Ridge Regularized Polynomial Model:\n")
cat("Best Lambda:", best_lambda, "\n")
cat("RMSE:", rmse(y_valid, pred_ridge), "\n")
cat("R²:", r2(y_valid, pred_ridge), "\n")
# Extract coefficients and print model equation
coef_ridge <- coef(cv_ridge, s = best_lambda)

# Extract coefficients by name
intercept <- coef_ridge[1]
fico_coef <- coef_ridge["fico", 1]
fico2_coef <- coef_ridge["fico2", 1]

cat("Ridge Regression Equation:\n")
cat(sprintf("int.rate = %.4f + %.4f * fico + %.6f * fico^2\n",
            intercept, fico_coef, fico2_coef))

#4D CODE#
# ---- GAM with Spline ----
model_gam <- gam(int.rate ~ s(fico), family = gaussian, data = train_set)
pred_gam <- predict(model_gam, newdata = valid_set)

# ---- Evaluation ----
cat("Linear Model:\n")
cat("RMSE:", rmse(valid_set$int.rate, pred_linear), "\n")
cat("R²:", r2(valid_set$int.rate, pred_linear), "\n\n")

cat("Polynomial Model:\n")
cat("RMSE:", rmse(valid_set$int.rate, pred_poly), "\n")
cat("R²:", r2(valid_set$int.rate, pred_poly), "\n\n")

cat("GAM Spline Model:\n")
cat("RMSE:", rmse(valid_set$int.rate, pred_gam), "\n")
cat("R²:", r2(valid_set$int.rate, pred_gam), "\n\n")

# ----Plot the GAM spline ----
plot(model_gam, se = TRUE, main = "Spline Fit for FICO")

# ---- Prepare Data for Plotting ----
plot_data <- data.frame(
  fico = valid_set$fico,
  actual = valid_set$int.rate,
  linear = pred_linear,
  poly = pred_poly,
  gam = pred_gam
)

# Reshape for ggplot long format
library(tidyr)
plot_long <- pivot_longer(
  plot_data,
  cols = c(linear, poly, gam),
  names_to = "model",
  values_to = "predicted"
)
library(ggplot2)

# Create a data frame with actual vs predicted (GAM only)
gam_plot_data <- data.frame(
  fico = valid_set$fico,
  actual = valid_set$int.rate,
  predicted = pred_gam
)

# Sort by fico for a smooth GAM line
gam_plot_data <- gam_plot_data[order(gam_plot_data$fico), ]

# Plot
ggplot(gam_plot_data, aes(x = fico)) +
  geom_point(aes(y = actual), color = "black", alpha = 0.5, size = 1.8) +
  geom_line(aes(y = predicted), color = "#009E73", size = 1.2) +
  labs(
    title = "GAM Model vs Actual Data",
    x = "FICO Score",
    y = "Interest Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  )

# ---- Plot with Color Key ----

ggplot() +
  geom_point(data = plot_data, aes(x = fico, y = actual), 
             alpha = 0.4, color = "black", size = 1.5, show.legend = TRUE) +
  geom_line(data = plot_long, aes(x = fico, y = predicted, color = model, linetype = model), size = 1.2) +
  scale_color_manual(
    name = "Model",
    values = c(
      "linear" = "#0072B2",  # blue
      "poly" = "#D55E00",    # red
      "gam" = "#009E73"      # green
    ),
    labels = c(
      "linear" = "Linear Model",
      "poly" = "Polynomial Model",
      "gam" = "GAM (Spline)"
    )
  ) +
  scale_linetype_manual(
    name = "Model",
    values = c(
      "linear" = "solid",
      "poly" = "dashed",
      "gam" = "twodash"
    ),
    labels = c(
      "linear" = "Linear Model",
      "poly" = "Polynomial Model",
      "gam" = "GAM (Spline)"
    )
  ) +
  labs(
    title = "Interest Rate Predictions vs FICO Score",
    subtitle = "Comparing Linear, Polynomial, and GAM Models",
    x = "FICO Score",
    y = "Interest Rate",
    caption = "Points = actual data, Lines = model predictions"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    legend.position = "right"
  )


results <- data.frame(
  Model = c("Linear", "Polynomial", "GAM"),
  RMSE_Train = c(
    rmse(train_set$int.rate, pred_train_linear),
    rmse(train_set$int.rate, pred_train_poly),
    rmse(train_set$int.rate, pred_train_gam)
  ),
  R2_Train = c(
    r2(train_set$int.rate, pred_train_linear),
    r2(train_set$int.rate, pred_train_poly),
    r2(train_set$int.rate, pred_train_gam)
  ),
  RMSE_Valid = c(
    rmse(valid_set$int.rate, pred_valid_linear),
    rmse(valid_set$int.rate, pred_valid_poly),
    rmse(valid_set$int.rate, pred_valid_gam)
  ),
  R2_Valid = c(
    r2(valid_set$int.rate, pred_valid_linear),
    r2(valid_set$int.rate, pred_valid_poly),
    r2(valid_set$int.rate, pred_valid_gam)
  )
)
results
#4E CODE###
# ---- Combine Predictions from Training and Validation ----

# Generate predictions on training set
train_set$pred_linear <- predict(model_linear, newdata = train_set)
train_set$pred_poly   <- predict(model_poly,   newdata = train_set)
train_set$pred_gam    <- predict(model_gam,    newdata = train_set)
train_set$set         <- "Training"

# Generate predictions on validation set
valid_set$pred_linear <- pred_linear
valid_set$pred_poly   <- pred_poly
valid_set$pred_gam    <- pred_gam
valid_set$set         <- "Validation"

# Combine sets
plot_df <- rbind(
  train_set[, c("fico", "int.rate", "pred_linear", "pred_poly", "pred_gam", "set")],
  valid_set[, c("fico", "int.rate", "pred_linear", "pred_poly", "pred_gam", "set")]
)

# Pivot to long format for ggplot
plot_long <- pivot_longer(
  plot_df,
  cols = c(pred_linear, pred_poly, pred_gam),
  names_to = "model",
  values_to = "prediction"
)

# Rename for cleaner labels
plot_long$model <- recode(plot_long$model,
                          pred_linear = "Linear Model",
                          pred_poly   = "Polynomial Model",
                          pred_gam    = "GAM (Spline)")

# ---- Final Bivariate Plot with Colorful Data Partitions ----
ggplot() +
  geom_point(data = plot_df, aes(x = fico, y = int.rate, color = set),
             alpha = 0.5, size = 1.5, show.legend = TRUE) +
  geom_line(data = plot_long, aes(x = fico, y = prediction, color = model, linetype = model), size = 1.2) +
  scale_color_manual(
    name = "Legend",
    values = c(
      "Training"        = "#999999",  # gray
      "Validation"      = "#E69F00",  # orange
      "Linear Model"     = "#0072B2", # blue
      "Polynomial Model" = "#D55E00", # red
      "GAM (Spline)"     = "#009E73"  # green
    ),
    breaks = c("Training", "Validation", "Linear Model", "Polynomial Model", "GAM (Spline)")
  ) +
  scale_linetype_manual(
    name = "Model",
    values = c(
      "Linear Model"     = "solid",
      "Polynomial Model" = "dashed",
      "GAM (Spline)"     = "twodash"
    )
  ) +
  labs(
    title = "Interest Rate Predictions vs FICO Score",
    subtitle = "Models Fit on Training and Validation Sets",
    x = "FICO Score",
    y = "Interest Rate",
    caption = "Points = actual data • Lines = model predictions"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

#4F CODE##
# Add polynomial term to test set
test_set$fico2 <- test_set$fico^2

# --- LINEAR ---
pred_train_linear <- predict(model_linear, newdata = train_set)
pred_valid_linear <- predict(model_linear, newdata = valid_set)
pred_test_linear  <- predict(model_linear, newdata = test_set)

# --- POLYNOMIAL ---
pred_train_poly <- predict(model_poly, newdata = train_set)
pred_valid_poly <- predict(model_poly, newdata = valid_set)
pred_test_poly  <- predict(model_poly, newdata = test_set)

# --- GAM SPLINE ---
pred_train_gam <- predict(model_gam, newdata = train_set)
pred_valid_gam <- predict(model_gam, newdata = valid_set)
pred_test_gam  <- predict(model_gam, newdata = test_set)

# --- RIDGE REGRESSION ---
X_train <- as.matrix(train_set[, c("fico", "fico2")])
X_valid <- as.matrix(valid_set[, c("fico", "fico2")])
X_test  <- as.matrix(test_set[,  c("fico", "fico2")])

y_train <- train_set$int.rate
y_valid <- valid_set$int.rate
y_test  <- test_set$int.rate

pred_train_ridge <- predict(cv_ridge, s = best_lambda, newx = X_train)
pred_valid_ridge <- predict(cv_ridge, s = best_lambda, newx = X_valid)
pred_test_ridge  <- predict(cv_ridge, s = best_lambda, newx = X_test)
library(ggplot2)

# Create a new data frame for ridge plot
ridge_plot_data <- data.frame(
  fico = valid_set$fico,
  actual = valid_set$int.rate,
  predicted = as.vector(pred_ridge)  # ensure prediction is a vector
)

# Sort by fico for smooth line
ridge_plot_data <- ridge_plot_data[order(ridge_plot_data$fico), ]

# Create the plot
ggplot(ridge_plot_data, aes(x = fico)) +
  geom_point(aes(y = actual), color = "black", alpha = 0.5, size = 1.8) +
  geom_line(aes(y = predicted), color = "#CC79A7", size = 1.2) +
  labs(
    title = "Ridge Regression vs Actual Data",
    subtitle = "Validation Set Predictions",
    x = "FICO Score",
    y = "Interest Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  )

results <- data.frame(
  Model = c("Linear", "Polynomial", "GAM Spline", "Ridge"),
  
  RMSE_Train = c(
    rmse(train_set$int.rate, pred_train_linear),
    rmse(train_set$int.rate, pred_train_poly),
    rmse(train_set$int.rate, pred_train_gam),
    rmse(y_train, pred_train_ridge)
  ),
  
  R2_Train = c(
    r2(train_set$int.rate, pred_train_linear),
    r2(train_set$int.rate, pred_train_poly),
    r2(train_set$int.rate, pred_train_gam),
    r2(y_train, pred_train_ridge)
  ),
  
  RMSE_Valid = c(
    rmse(valid_set$int.rate, pred_valid_linear),
    rmse(valid_set$int.rate, pred_valid_poly),
    rmse(valid_set$int.rate, pred_valid_gam),
    rmse(y_valid, pred_valid_ridge)
  ),
  
  R2_Valid = c(
    r2(valid_set$int.rate, pred_valid_linear),
    r2(valid_set$int.rate, pred_valid_poly),
    r2(valid_set$int.rate, pred_valid_gam),
    r2(y_valid, pred_valid_ridge)
  )
)

library(knitr)
kable(results, digits = 4, caption = "In-Sample and Out-of-Sample Performance (Training & Validation Sets)")
cat("Final Evaluation on Uncontaminated Test Set (GAM Spline):\n")
cat("RMSE:", rmse(test_set$int.rate, pred_test_gam), "\n")
cat("R²:", r2(test_set$int.rate, pred_test_gam), "\n")


#MULTIVARIATE REGRESSION MODELLING#####
#5A CODE#
# Fit model on training data
model_multi <- lm(int.rate ~ fico + dti + log.annual.inc + purpose + not.fully.paid,
                    data = train_set)
# Summary of the model
summary(model_multi)
# Predict on training and validation sets
pred_train_multi <- predict(model_multi, newdata = train_set)
pred_valid_multi <- predict(model_multi, newdata = valid_set)

# RMSE and R² functions (already defined)
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2   <- function(actual, pred) cor(actual, pred)^2

# Training performance
cat("Training RMSE:", rmse(train_set$int.rate, pred_train_multi), "\n")
cat("Training R²:", r2(train_set$int.rate, pred_train_multi), "\n\n")

# Validation performance
cat("Validation RMSE:", rmse(valid_set$int.rate, pred_valid_multi), "\n")
cat("Validation R²:", r2(valid_set$int.rate, pred_valid_multi), "\n")


# Create a plot-friendly data frame
multi_plot_data <- data.frame(
  fico = valid_set$fico,
  actual = valid_set$int.rate,
  predicted = pred_valid_multi
)

# Sort by FICO for smoother lines
multi_plot_data <- multi_plot_data[order(multi_plot_data$fico), ]

# Create the plot
ggplot(multi_plot_data, aes(x = fico)) +
  geom_point(aes(y = actual), color = "black", alpha = 0.5, size = 1.8) +
  geom_line(aes(y = predicted), color = "#0072B2", size = 1.2) +
  labs(
    title = "Multivariate Linear Regression vs Actual Data",
    x = "FICO Score",
    y = "Interest Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  )


#5B CODE#
# Prepare matrix input (X) with dummy variables
#use model.matrix() to handle categorical encoding
X_train <- model.matrix(int.rate ~ fico + dti + log.annual.inc + purpose + not.fully.paid, data = train_set)[, -1]
y_train <- train_set$int.rate

X_valid <- model.matrix(int.rate ~ fico + dti + log.annual.inc + purpose + not.fully.paid, data = valid_set)[, -1]
y_valid <- valid_set$int.rate
set.seed(123)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)  # Ridge = alpha 0
best_lambda_ridge <- cv_ridge$lambda.min

# Create a plot-friendly data frame
ridge_multi_plot_data <- data.frame(
  fico = valid_set$fico,
  actual = y_valid,
  predicted = as.vector(pred_valid_ridge)  # convert to numeric vector if needed
)

# Sort by fico for smooth curve
ridge_multi_plot_data <- ridge_multi_plot_data[order(ridge_multi_plot_data$fico), ]

# Plot actual vs predicted
ggplot(ridge_multi_plot_data, aes(x = fico)) +
  geom_point(aes(y = actual), color = "black", alpha = 0.5, size = 1.8) +
  geom_line(aes(y = predicted), color = "#CC79A7", size = 1.2) +
  labs(
    title = "Ridge Regression (Multivariate) vs Actual Data",
    x = "FICO Score",
    y = "Interest Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  )


# Predict on training and validation
pred_train_ridge <- predict(cv_ridge, s = best_lambda_ridge, newx = X_train)
pred_valid_ridge <- predict(cv_ridge, s = best_lambda_ridge, newx = X_valid)
cat("Ridge Regression:\n")
cat("Best Lambda:", best_lambda_ridge, "\n")
cat("Training RMSE:", rmse(y_train, pred_train_ridge), "\n")
cat("Training R²:", r2(y_train, pred_train_ridge), "\n\n")
cat("Validation RMSE:", rmse(y_valid, pred_valid_ridge), "\n")
cat("Validation R²:", r2(y_valid, pred_valid_ridge), "\n")
# comparison table
pred_train_multi <- predict(model_multi, newdata = train_set)
pred_valid_multi <- predict(model_multi, newdata = valid_set)
pred_train_ridge <- as.vector(pred_train_ridge)
pred_valid_ridge <- as.vector(pred_valid_ridge)
results_compare <- data.frame(
  Model = c("Multivariate Linear", "Ridge Regression"),
  
  RMSE_Train = c(
    rmse(train_set$int.rate, pred_train_multi),
    rmse(y_train, pred_train_ridge)
  ),
  
  R2_Train = c(
    r2(train_set$int.rate, pred_train_multi),
    r2(y_train, pred_train_ridge)
  ),
  
  RMSE_Valid = c(
    rmse(valid_set$int.rate, pred_valid_multi),
    rmse(y_valid, pred_valid_ridge)
  ),
  
  R2_Valid = c(
    r2(valid_set$int.rate, pred_valid_multi),
    r2(y_valid, pred_valid_ridge)
  )
)

kable(results_compare, digits = 4, caption = "Benchmark: Multivariate Linear vs Ridge-Regularized Model")

#5C CODE###
# Add nonlinear terms
train_set$fico2 <- train_set$fico^2
valid_set$fico2 <- valid_set$fico^2

train_set$log.dti <- log(train_set$dti + 1)
valid_set$log.dti <- log(valid_set$dti + 1)
model_nonlinear_multi <- lm(int.rate ~ fico + fico2 + log.dti +
                                log.annual.inc + purpose + not.fully.paid,
                              data = train_set)

summary(model_nonlinear_multi)
pred_train_nl_multi <- predict(model_nonlinear_multi, newdata = train_set)
pred_valid_nl_multi <- predict(model_nonlinear_multi, newdata = valid_set)
pred_train_ridge <- as.vector(pred_train_ridge)
pred_valid_ridge <- as.vector(pred_valid_ridge)
results_nonlinear_compare <- data.frame(
  Model = c("Linear", "Ridge Regression", "Nonlinear (No Reg)"),
  
  RMSE_Train = c(
    rmse(train_set$int.rate, pred_train_multi),
    rmse(y_train, pred_train_ridge),
    rmse(train_set$int.rate, pred_train_nl_multi)
  ),
  
  R2_Train = c(
    r2(train_set$int.rate, pred_train_multi),
    r2(y_train, pred_train_ridge),
    r2(train_set$int.rate, pred_train_nl_multi)
  ),
  
  RMSE_Valid = c(
    rmse(valid_set$int.rate, pred_valid_multi),
    rmse(y_valid, pred_valid_ridge),
    rmse(valid_set$int.rate, pred_valid_nl_multi)
  ),
  
  R2_Valid = c(
    r2(valid_set$int.rate, pred_valid_multi),
    r2(y_valid, pred_valid_ridge),
    r2(valid_set$int.rate, pred_valid_nl_multi)
  )
)

kable(results_nonlinear_compare, digits = 4,
      caption = "Benchmark: Linear vs Ridge vs Nonlinear (Unregularized)")

#5D CODE####
library(e1071)
# Ensure necessary variables are added
train_set$fico2 <- train_set$fico^2
valid_set$fico2 <- valid_set$fico^2
train_set$log.dti <- log(train_set$dti + 1)
valid_set$log.dti <- log(valid_set$dti + 1)
# Fit SVM regression model
svm_model <- svm(int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
                 data = train_set,
                 type = "eps-regression",  # standard for regression
                 kernel = "radial")        # RBF kernel (default)
# Predictions
pred_train_svm <- predict(svm_model, newdata = train_set)
pred_valid_svm <- predict(svm_model, newdata = valid_set)
train_rmse_svm <- rmse(train_set$int.rate, pred_train_svm)
train_r2_svm   <- r2(train_set$int.rate, pred_train_svm)

valid_rmse_svm <- rmse(valid_set$int.rate, pred_valid_svm)
valid_r2_svm   <- r2(valid_set$int.rate, pred_valid_svm)

# Performance
cat("SVM Regression:\n")
cat("Training RMSE:", rmse(train_set$int.rate, pred_train_svm), "\n")
cat("Training R²:", r2(train_set$int.rate, pred_train_svm), "\n\n")

cat("Validation RMSE:", rmse(valid_set$int.rate, pred_valid_svm), "\n")
cat("Validation R²:", r2(valid_set$int.rate, pred_valid_svm), "\n")
# Create the table
svm_untuned_results <- data.frame(
  Set = c("Training", "Validation"),
  RMSE = c(train_rmse_svm, valid_rmse_svm),
  R2   = c(train_r2_svm, valid_r2_svm)
)

# Display nicely
kable(svm_untuned_results, digits = 4, caption = "Untuned SVM Model Performance")

set.seed(123)
#TUNE
svm_tuned <- tune(
  svm,
  int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
  data = train_set,
  ranges = list(cost = c(1, 10), gamma = c(0.1, 1)),
  tunecontrol = tune.control(cross = 5)  # 5-fold instead of 10
)

# Best model from tuning
best_svm <- svm_tuned$best.model

# Predictions from tuned model
pred_train_svm_tuned <- predict(best_svm, newdata = train_set)
pred_valid_svm_tuned <- predict(best_svm, newdata = valid_set)

# Ensure functions are defined
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2   <- function(actual, pred) cor(actual, pred)^2

# Calculate performance
train_rmse_svm_tuned <- rmse(train_set$int.rate, pred_train_svm_tuned)
train_r2_svm_tuned   <- r2(train_set$int.rate, pred_train_svm_tuned)
valid_rmse_svm_tuned <- rmse(valid_set$int.rate, pred_valid_svm_tuned)
valid_r2_svm_tuned   <- r2(valid_set$int.rate, pred_valid_svm_tuned)

# Print results
cat("Tuned SVM Performance:\n")
cat("Training RMSE:", train_rmse_svm_tuned, "\n")
cat("Training R²:", train_r2_svm_tuned, "\n")
cat("Validation RMSE:", valid_rmse_svm_tuned, "\n")
cat("Validation R²:", valid_r2_svm_tuned, "\n")
# Create the table
svm_results <- data.frame(
  Set = c("Training", "Validation"),
  RMSE = c(train_rmse_svm_tuned, valid_rmse_svm_tuned),
  R2   = c(train_r2_svm_tuned, valid_r2_svm_tuned)
)

# Display nicely
kable(svm_results, digits = 4, caption = "Tuned SVM Model Performance")

#5E CODE### REGRESSION TREES###
library(rpart)
library(rpart.plot)
# Ensure nonlinear features are added
train_set$fico2 <- train_set$fico^2
valid_set$fico2 <- valid_set$fico^2
train_set$log.dti <- log(train_set$dti + 1)
valid_set$log.dti <- log(valid_set$dti + 1)

# Build the regression tree
tree_model <- rpart(int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
                    data = train_set,
                    method = "anova",
                    control = rpart.control(cp = 0.01))  # You can tweak cp to control size
rpart.plot(tree_model, type = 2, extra = 101, fallen.leaves = TRUE, main = "Regression Tree for Interest Rate")
# Predict
pred_train_tree <- predict(tree_model, newdata = train_set)
pred_valid_tree <- predict(tree_model, newdata = valid_set)

# Define metrics
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2 <- function(actual, pred) cor(actual, pred)^2

# Compute
train_rmse_tree <- rmse(train_set$int.rate, pred_train_tree)
train_r2_tree <- r2(train_set$int.rate, pred_train_tree)

valid_rmse_tree <- rmse(valid_set$int.rate, pred_valid_tree)
valid_r2_tree <- r2(valid_set$int.rate, pred_valid_tree)

# Print
cat("Regression Tree Performance:\n")
cat("Training RMSE:", train_rmse_tree, "\n")
cat("Training R²:", train_r2_tree, "\n")
cat("Validation RMSE:", valid_rmse_tree, "\n")
cat("Validation R²:", valid_r2_tree, "\n")
# Create the table
tree_results <- data.frame(
  Set = c("Training", "Validation"),
  RMSE = c(train_rmse_tree, valid_rmse_tree),
  R2   = c(train_r2_tree, valid_r2_tree)
)
library(knitr)
# Display the table
kable(tree_results, digits = 4, caption = "Regression Tree Model Performance")

##TUNING##
printcp(tree_model)   # View cp table

# Choose best cp
best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]

# Prune tree
pruned_tree <- prune(tree_model, cp = best_cp)

# Predict again
pred_train_pruned <- predict(pruned_tree, newdata = train_set) 
pred_valid_pruned <- predict(pruned_tree, newdata = valid_set)
valid_rmse_pruned <- rmse(valid_set$int.rate, pred_valid_pruned)
valid_r2_pruned <- r2(valid_set$int.rate, pred_valid_pruned)
train_rmse_pruned <- rmse(train_set$int.rate, pred_train_pruned)
train_r2_pruned <- r2(train_set$int.rate, pred_train_pruned)     

cat("Pruned Tree Validation RMSE:", valid_rmse_pruned, "\n")
cat("Pruned Tree Validation R²:", valid_r2_pruned, "\n")
cat("Pruned Tree Training RMSE:", train_rmse_pruned, "\n") 
cat("Prunes Tree Training R²:", train_r2_pruned, "\n")     
# Create the table
pruned_tree_results <- data.frame(
  Set = c("Training", "Validation"),
  RMSE = c(train_rmse_pruned, valid_rmse_pruned),
  R2   = c(train_r2_pruned, valid_r2_pruned)
)

# Display the table
kable(pruned_tree_results, digits = 4, caption = "Pruned Regression Tree Model Performance")
library(rpart.plot)

# Plot the pruned tree
rpart.plot(
  pruned_tree,
  type = 2,               # label all nodes
  extra = 101,            # show fitted value + % of obs
  fallen.leaves = TRUE,   # layout neatly
  main = "Pruned Regression Tree for Interest Rate",
  box.palette = "BuGn",   # color gradient for output value
  tweak = 1.2             # sizing adjustment (optional)
)


#5F TREES BASED ENSEMBLE MODEL##
library(randomForest)
# Make sure features exist
train_set$fico2 <- train_set$fico^2
valid_set$fico2 <- valid_set$fico^2
train_set$log.dti <- log(train_set$dti + 1)
valid_set$log.dti <- log(valid_set$dti + 1)

# Fit random forest
set.seed(123)
rf_model <- randomForest(
  int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
  data = train_set,
  ntree = 500,           # number of trees
  mtry = 3,              # number of predictors tried at each split
  importance = TRUE
)
# Predict
pred_train_rf <- predict(rf_model, newdata = train_set)
pred_valid_rf <- predict(rf_model, newdata = valid_set)

# Performance metrics
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2   <- function(actual, pred) cor(actual, pred)^2

train_rmse_rf <- rmse(train_set$int.rate, pred_train_rf)
train_r2_rf   <- r2(train_set$int.rate, pred_train_rf)

valid_rmse_rf <- rmse(valid_set$int.rate, pred_valid_rf)
valid_r2_rf   <- r2(valid_set$int.rate, pred_valid_rf)

# Print results
cat("Random Forest Performance:\n")
cat("Training RMSE:", train_rmse_rf, "\n")
cat("Training R²:", train_r2_rf, "\n")
cat("Validation RMSE:", valid_rmse_rf, "\n")
cat("Validation R²:", valid_r2_rf, "\n")
library(caret)
# Create the table
rf_results <- data.frame(
  Set = c("Training", "Validation"),
  RMSE = c(train_rmse_rf, valid_rmse_rf),
  R2   = c(train_r2_rf, valid_r2_rf)
)

# Display the table
kable(rf_results, digits = 4, caption = "Random Forest Model Performance")
library(ggplot2)

# Create a data frame for plotting
rf_plot_data <- data.frame(
  fico = valid_set$fico,
  actual = valid_set$int.rate,
  predicted = pred_valid_rf
)

# Sort by FICO score for a clean line
rf_plot_data <- rf_plot_data[order(rf_plot_data$fico), ]

# Plot
ggplot(rf_plot_data, aes(x = fico)) +
  geom_point(aes(y = actual), color = "black", alpha = 0.5, size = 1.8) +
  geom_line(aes(y = predicted), color = "#E69F00", size = 1.2) +
  labs(
    title = "Random Forest Predictions vs Actual Interest Rates",
    x = "FICO Score",
    y = "Interest Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  )

# Create training control
control <- trainControl(method = "cv", number = 3)

# Train RF with tuning
set.seed(123)
rf_tuned <- train(
  int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
  data = train_set,
  method = "rf",
  trControl = control,
  tuneLength = 2  # number of mtry values to try
)

# Best model and predictions

pred_train_rf_tuned <- predict(rf_tuned, newdata = train_set) 
best_rf <- rf_tuned$finalModel
pred_valid_rf_tuned <- predict(rf_tuned, newdata = valid_set)

# Performance
train_rmse_rf_tuned <- rmse(train_set$int.rate, pred_train_rf_tuned)  
train_r2_rf_tuned <- r2(train_set$int.rate, pred_train_rf_tuned)
valid_rmse_rf_tuned <- rmse(valid_set$int.rate, pred_valid_rf_tuned)
valid_r2_rf_tuned   <- r2(valid_set$int.rate, pred_valid_rf_tuned)

cat("Tuned Random Forest Validation RMSE:", valid_rmse_rf_tuned, "\n")
cat("Tuned Random Forest Validation R²:", valid_r2_rf_tuned, "\n")
cat("Tuned Random Forest Training RMSE:", train_rmse_rf_tuned, "\n")  
cat("Tuned Random Forest Training R²:", train_r2_rf_tuned, "\n")    
# Create the table
rf_tuned_results <- data.frame(
  Set = c("Training", "Validation"),
  RMSE = c(train_rmse_rf_tuned, valid_rmse_rf_tuned),
  R2   = c(train_r2_rf_tuned, valid_r2_rf_tuned)
)

# Display the table
kable(rf_tuned_results, digits = 4, caption = "Tuned Random Forest Model Performance")

library(ggplot2)

# Create plot data frame
rf_tuned_plot_data <- data.frame(
  fico = valid_set$fico,
  actual = valid_set$int.rate,
  predicted = pred_valid_rf_tuned
)

# Sort for smooth line
rf_tuned_plot_data <- rf_tuned_plot_data[order(rf_tuned_plot_data$fico), ]

# Plot
ggplot(rf_tuned_plot_data, aes(x = fico)) +
  geom_point(aes(y = actual), color = "black", alpha = 0.5, size = 1.8) +
  geom_line(aes(y = predicted), color = "#56B4E9", size = 1.2) +
  labs(
    title = "Tuned Random Forest Predictions vs Actual Interest Rates",
    x = "FICO Score",
    y = "Interest Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold")
  )


#5G SUMMARY TABLE W RESULTS##
# Ensure all predictions are numeric vectors
pred_train_ridge <- as.vector(pred_train_ridge)
pred_valid_ridge <- as.vector(pred_valid_ridge)

# Build final comparison table
model_results <- data.frame(
  Model = c("Multivariate Linear", "Ridge Regression", "Nonlinear (No Reg)",
            "SVM (Tuned)", "Regression Tree", "Pruned Tree", "Random Forest", "Tuned Random Forest"),
  
  RMSE_Train = c(
    rmse(train_set$int.rate, pred_train_multi),
    rmse(y_train, pred_train_ridge),
    rmse(train_set$int.rate, pred_train_nl_multi),
    rmse(train_set$int.rate, pred_train_svm_tuned),
    rmse(train_set$int.rate, pred_train_tree),
    rmse(train_set$int.rate, pred_train_pruned),
    rmse(train_set$int.rate, pred_train_rf),
    rmse(train_set$int.rate, pred_train_rf_tuned)
  ),
  
  R2_Train = c(
    r2(train_set$int.rate, pred_train_multi),
    r2(y_train, pred_train_ridge),
    r2(train_set$int.rate, pred_train_nl_multi),
    r2(train_set$int.rate, pred_train_svm_tuned),
    r2(train_set$int.rate, pred_train_tree),
    r2(train_set$int.rate, pred_train_pruned),
    r2(train_set$int.rate, pred_train_rf),
    r2(train_set$int.rate, pred_train_rf_tuned)
  ),
  
  RMSE_Valid = c(
    rmse(valid_set$int.rate, pred_valid_multi),
    rmse(valid_set$int.rate, pred_valid_ridge),
    rmse(valid_set$int.rate, pred_valid_nl_multi),
    rmse(valid_set$int.rate, pred_valid_svm_tuned),
    rmse(valid_set$int.rate, pred_valid_tree),
    rmse(valid_set$int.rate, pred_valid_pruned),
    rmse(valid_set$int.rate, pred_valid_rf),
    rmse(valid_set$int.rate, pred_valid_rf_tuned)
  ),
  
  R2_Valid = c(
    r2(valid_set$int.rate, pred_valid_multi),
    r2(valid_set$int.rate, pred_valid_ridge),
    r2(valid_set$int.rate, pred_valid_nl_multi),
    r2(valid_set$int.rate, pred_valid_svm_tuned),
    r2(valid_set$int.rate, pred_valid_tree),
    r2(valid_set$int.rate, pred_valid_pruned),
    r2(valid_set$int.rate, pred_valid_rf),
    r2(valid_set$int.rate, pred_valid_rf_tuned)
  )
)


# Display table
library(knitr)
kable(results_final, digits = 4, caption = "Final Model Benchmark: Training and Validation Performance")



# 5E: REGRESSION TREE
library(rpart)
library(rpart.plot)

# Create polynomial and log terms
train_set$fico2 <- train_set$fico^2
valid_set$fico2 <- valid_set$fico^2
train_set$log.dti <- log(train_set$dti + 1)
valid_set$log.dti <- log(valid_set$dti + 1)

# Fit regression tree
tree_model <- rpart(int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
                    data = train_set,
                    method = "anova",
                    control = rpart.control(cp = 0.01))

# Plot tree
rpart.plot(tree_model, type = 2, extra = 101, fallen.leaves = TRUE, main = "Regression Tree for Interest Rate")

# Predictions
pred_train_tree <- predict(tree_model, newdata = train_set)
pred_valid_tree <- predict(tree_model, newdata = valid_set)

# Metrics
train_rmse_tree <- rmse(train_set$int.rate, pred_train_tree)
train_r2_tree   <- r2(train_set$int.rate, pred_train_tree)
valid_rmse_tree <- rmse(valid_set$int.rate, pred_valid_tree)
valid_r2_tree   <- r2(valid_set$int.rate, pred_valid_tree)

# Print
cat("Regression Tree:\n")
cat("Train RMSE:", train_rmse_tree, "\n")
cat("Train R²:", train_r2_tree, "\n")
cat("Valid RMSE:", valid_rmse_tree, "\n")
cat("Valid R²:", valid_r2_tree, "\n")

# Prune tree
printcp(tree_model)
best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
pruned_tree <- prune(tree_model, cp = best_cp)

# Pruned predictions
pred_train_pruned <- predict(pruned_tree, newdata = train_set)
pred_valid_pruned <- predict(pruned_tree, newdata = valid_set)
train_rmse_pruned <- rmse(train_set$int.rate, pred_train_pruned)
train_r2_pruned   <- r2(train_set$int.rate, pred_train_pruned)
valid_rmse_pruned <- rmse(valid_set$int.rate, pred_valid_pruned)
valid_r2_pruned   <- r2(valid_set$int.rate, pred_valid_pruned)

cat("Pruned Tree:\n")
cat("Train RMSE:", train_rmse_pruned, "\n")
cat("Train R²:", train_r2_pruned, "\n")
cat("Valid RMSE:", valid_rmse_pruned, "\n")
cat("Valid R²:", valid_r2_pruned, "\n")


# --- 5F: RANDOM FOREST + TUNING ---
library(randomForest)
library(caret)

# Fit untuned random forest
set.seed(123)
rf_model <- randomForest(int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
                         data = train_set, ntree = 500, mtry = 3, importance = TRUE)

pred_train_rf <- predict(rf_model, newdata = train_set)
pred_valid_rf <- predict(rf_model, newdata = valid_set)
train_rmse_rf <- rmse(train_set$int.rate, pred_train_rf)
train_r2_rf   <- r2(train_set$int.rate, pred_train_rf)
valid_rmse_rf <- rmse(valid_set$int.rate, pred_valid_rf)
valid_r2_rf   <- r2(valid_set$int.rate, pred_valid_rf)

# Tune RF using caret
control <- trainControl(method = "cv", number = 5)
set.seed(123)
rf_tuned <- train(int.rate ~ fico + fico2 + log.dti + log.annual.inc + purpose + not.fully.paid,
                  data = train_set, method = "rf", trControl = control, tuneLength = 5)

# Best tuned RF model
pred_train_rf_tuned <- predict(rf_tuned, newdata = train_set)
pred_valid_rf_tuned <- predict(rf_tuned, newdata = valid_set)
train_rmse_rf_tuned <- rmse(train_set$int.rate, pred_train_rf_tuned)
train_r2_rf_tuned   <- r2(train_set$int.rate, pred_train_rf_tuned)
valid_rmse_rf_tuned <- rmse(valid_set$int.rate, pred_valid_rf_tuned)
valid_r2_rf_tuned   <- r2(valid_set$int.rate, pred_valid_rf_tuned)

cat("Tuned Random Forest:\n")
cat("Train RMSE:", train_rmse_rf_tuned, "\n")
cat("Train R²:", train_r2_rf_tuned, "\n")
cat("Valid RMSE:", valid_rmse_rf_tuned, "\n")
cat("Valid R²:", valid_r2_rf_tuned, "\n")


##5G CODE###FINAL TABLE
# Ensure predictions are vectors
pred_train_ridge <- as.vector(pred_train_ridge)
pred_valid_ridge <- as.vector(pred_valid_ridge)

# Add Linear, Polynomial, Ridge, and GAM to results_final
results_final <- data.frame(
  Model = c(
    "Linear (Bivariate)",
    "Polynomial (Bivariate)",
    "Ridge (Bivariate)",
    "GAM (Spline)",
    "Multivariate Linear",
    "Ridge Regression",
    "Nonlinear (No Reg)",
    "SVM (Tuned)",
    "Regression Tree",
    "Pruned Tree",
    "Random Forest",
    "Tuned Random Forest"
  ),
  
  RMSE_Train = c(
    rmse(train_set$int.rate, pred_train_linear),
    rmse(train_set$int.rate, pred_train_poly),
    rmse(train_set$int.rate, pred_train_ridge),
    rmse(train_set$int.rate, pred_train_gam),
    rmse(train_set$int.rate, pred_train_multi),
    rmse(y_train, pred_train_ridge),
    rmse(train_set$int.rate, pred_train_nl_multi),
    rmse(train_set$int.rate, pred_train_svm_tuned),
    rmse(train_set$int.rate, pred_train_tree),
    rmse(train_set$int.rate, pred_train_pruned),
    rmse(train_set$int.rate, pred_train_rf),
    rmse(train_set$int.rate, pred_train_rf_tuned)
  ),
  
  R2_Train = c(
    r2(train_set$int.rate, pred_train_linear),
    r2(train_set$int.rate, pred_train_poly),
    r2(train_set$int.rate, pred_train_ridge),
    r2(train_set$int.rate, pred_train_gam),
    r2(train_set$int.rate, pred_train_multi),
    r2(y_train, pred_train_ridge),
    r2(train_set$int.rate, pred_train_nl_multi),
    r2(train_set$int.rate, pred_train_svm_tuned),
    r2(train_set$int.rate, pred_train_tree),
    r2(train_set$int.rate, pred_train_pruned),
    r2(train_set$int.rate, pred_train_rf),
    r2(train_set$int.rate, pred_train_rf_tuned)
  ),
  
  RMSE_Valid = c(
    rmse(valid_set$int.rate, pred_valid_linear),
    rmse(valid_set$int.rate, pred_valid_poly),
    rmse(valid_set$int.rate, pred_valid_ridge),
    rmse(valid_set$int.rate, pred_valid_gam),
    rmse(valid_set$int.rate, pred_valid_multi),
    rmse(y_valid, pred_valid_ridge),
    rmse(valid_set$int.rate, pred_valid_nl_multi),
    rmse(valid_set$int.rate, pred_valid_svm_tuned),
    rmse(valid_set$int.rate, pred_valid_tree),
    rmse(valid_set$int.rate, pred_valid_pruned),
    rmse(valid_set$int.rate, pred_valid_rf),
    rmse(valid_set$int.rate, pred_valid_rf_tuned)
  ),
  
  R2_Valid = c(
    r2(valid_set$int.rate, pred_valid_linear),
    r2(valid_set$int.rate, pred_valid_poly),
    r2(valid_set$int.rate, pred_valid_ridge),
    r2(valid_set$int.rate, pred_valid_gam),
    r2(valid_set$int.rate, pred_valid_multi),
    r2(y_valid, pred_valid_ridge),
    r2(valid_set$int.rate, pred_valid_nl_multi),
    r2(valid_set$int.rate, pred_valid_svm_tuned),
    r2(valid_set$int.rate, pred_valid_tree),
    r2(valid_set$int.rate, pred_valid_pruned),
    r2(valid_set$int.rate, pred_valid_rf),
    r2(valid_set$int.rate, pred_valid_rf_tuned)
  )
)

# Display the final model comparison table
library(knitr)
kable(results_final, digits = 4, caption = "Final Model Benchmark: Training and Validation Performance")
library(dplyr)
library(knitr)
library(kableExtra)

# First, store min/max values with NAs removed
min_rmse_train <- min(results_final$RMSE_Train, na.rm = TRUE)
max_r2_train   <- max(results_final$R2_Train, na.rm = TRUE)
min_rmse_valid <- min(results_final$RMSE_Valid, na.rm = TRUE)
max_r2_valid   <- max(results_final$R2_Valid, na.rm = TRUE)

# Create a fully character-safe version of the table
highlighted_table <- results_final %>%
  mutate(
    RMSE_Train = ifelse(
      is.na(RMSE_Train),
      "N/A",
      ifelse(RMSE_Train == min_rmse_train,
             cell_spec(sprintf("%.4f", RMSE_Train), color = "white", background = "#1b7837", format = "html"),
             sprintf("%.4f", RMSE_Train))
    ),
    R2_Train = ifelse(
      is.na(R2_Train),
      "N/A",
      ifelse(R2_Train == max_r2_train,
             cell_spec(sprintf("%.4f", R2_Train), color = "white", background = "#1b7837", format = "html"),
             sprintf("%.4f", R2_Train))
    ),
    RMSE_Valid = ifelse(
      is.na(RMSE_Valid),
      "N/A",
      ifelse(RMSE_Valid == min_rmse_valid,
             cell_spec(sprintf("%.4f", RMSE_Valid), color = "white", background = "#2166ac", format = "html"),
             sprintf("%.4f", RMSE_Valid))
    ),
    R2_Valid = ifelse(
      is.na(R2_Valid),
      "N/A",
      ifelse(R2_Valid == max_r2_valid,
             cell_spec(sprintf("%.4f", R2_Valid), color = "white", background = "#2166ac", format = "html"),
             sprintf("%.4f", R2_Valid))
    )
  )

# Render styled table
kable(highlighted_table, escape = FALSE, format = "html",
      caption = "Final Model Benchmark with Highlights") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed"))
