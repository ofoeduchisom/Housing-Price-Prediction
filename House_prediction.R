
```{r}
# Get the current working directory
getwd()
```

```{r}
# Set the working directory to the folder containing your CSV file and load the csv file
setwd("C:/Users/Admin/Downloads")
Data <- read.csv("Housing Data_Same Region.csv")
Data
```


```{r}
install.packages("dplyr")  # For data manipulation; part of the tidyverse
install.packages("caret")  # For machine learning algorithms and model training
install.packages("corrplot")  # For visualizing correlation matrices
install.packages("ggplot2")  # For creating high-quality data visualizations
install.packages("randomForest") 
install.packages("varImp")
```

```{r}
library(caret)
library(ggplot2)
library(dplyr)
library(corrplot)
library(e1071)
library(rpart)
library(randomForest)
library(varImp)
```

```{r}
# Display first few rows of data frame
head(Data)
```

```{r}
# View data type of each column
str(Data)
```

```{r}
#Visualising Outliers for SALE_PRC          
boxplot(Data$SALE_PRC, 
        main = "Box Plot for Sales Price", 
        ylab = "Sales Price", 
        col = "lightblue", 
        border = "darkblue",
        horizontal = FALSE)
```

```{r}
#Visualising Outliers for TOT_LVG_AREA              
boxplot(Data$TOT_LVG_AREA, 
        main = "Box Plot for total living average", 
        ylab = "Total living average", 
        col = "lightblue", 
        border = "darkblue", 
        horizontal = FALSE)
```

```{r}
# Remove specific columns from the data frame
Data <- subset(Data, select = -c(LATITUDE, LONGITUDE, PARCELNO, avno60plus))
```

```{r}
# Set SALE_PRC column as the last column
Data <- Data[, c(setdiff(names(Data), "SALE_PRC"), "SALE_PRC")]
Data
```

```{r}
# Check for duplicates in rows
sum(duplicated(Data))
```

```{r}
# check for missing rows with NA values
na_summary <- colSums(is.na(Data))
print(na_summary)
```

```{r}
# Checking which columns have zero values
zero_values <- sapply(Data, function(x) sum(x == 0))
print(zero_values) 
```

```{r}
# Replace zeros with NA
Data$SPEC_FEAT_VAL[Data$SPEC_FEAT_VAL == 0] <- NA
Data$age[Data$age == 0] <- NA
Data$WATER_DIST [Data$WATER_DIST  == 0] <- NA
```

```{r}
# Impute missing values
Data$SPEC_FEAT_VAL[is.na(Data$SPEC_FEAT_VAL)] <- median(Data$SPEC_FEAT_VAL, na.rm = TRUE)
Data$age[is.na(Data$age)] <- median(Data$age, na.rm = TRUE)
Data$WATER_DIST[is.na(Data$WATER_DIST)] <- median(Data$WATER_DIST, na.rm = TRUE)
```

```{r}
# Check if there are any remaining NAs
sum(is.na(Data))
```

```{r}
# Compute correlation matrix for numeric columns
cor_matrix <- cor(Data[, sapply(Data, is.numeric)])

# Correlation of all features with the target variable SALE_PRC
cor_with_target <- cor_matrix[, "SALE_PRC"]

# View correlations
print(cor_with_target)
```
```{r}
# Visualize the correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper")
```

```{r}
# Select most relevant features based on correlation
selected_features <- c("TOT_LVG_AREA", "LND_SQFOOT", "SPEC_FEAT_VAL", "structure_quality")

# Create a subset of the original dataset to keep only selected features and the target variable
model_data <- Data[, c(selected_features, "SALE_PRC")]
```


```{r}
set.seed(123)

# Split the data into training and test sets (80% training, 20% test)
trainIndex <- createDataPartition(model_data$SALE_PRC, p = 0.8, list = FALSE)

# Create training and test data
train_data <- model_data[trainIndex, ]
test_data <- model_data[-trainIndex, ]
```


```{r}
# Get the dimensions (number of rows and columns) of the training and test dataset
dim(train_data)
dim(test_data)
```

```{r}
# Linear Regression Model
# Fit model on the training data
lm_model <- lm(SALE_PRC ~ ., data = train_data)

# Make prediction on the test set
lm_model_predictions <- predict(lm_model, newdata=test_data)

# Calculate performance metrics
lm_performance <- postResample(pred = lm_model_predictions, obs = test_data$SALE_PRC)

# print performance metric
print(lm_performance)
```


```{r}
# Support Vector Regression (SVR) Model
svr_model <- svm(SALE_PRC ~ ., data = train_data, kernel="linear")

# Make prediction on the test set 
svr_model_predictions <- predict(svr_model, newdata=test_data)

# Calculate performance metrics
svr_performance <- postResample(pred = svr_model_predictions, obs = test_data$SALE_PRC)

# print performance metric
print(svr_performance)
```

```{r}
# Decision Tree Model
dt_model <- rpart(SALE_PRC ~ ., data = train_data)

# Make prediction on the test set
dt_model_prediction <- predict(dt_model, newdata=test_data)

# Calculate performance metrics
dt_performance <- postResample(pred = dt_model_prediction, obs = test_data$SALE_PRC)

# print performance metric
print(dt_performance)
```

```{r}
# Random Forest Model 
# Fit model on the training data when n=100
rf_model_n100 <- randomForest(SALE_PRC ~., data = train_data, ntree=100)

# Make prediction on the test set
rf_model_prediction_n100 <- predict(rf_model_n100, newdata = test_data)

# Calculate performance metrics
rf_performance_n100 <- postResample(pred = rf_model_prediction_n100, obs = test_data$SALE_PRC)

# print performance metric
print(rf_performance_n100)
```

```{r}
# Fit model on the training data when n=200
rf_model_n200 <- randomForest(SALE_PRC ~ ., data = train_data, ntree=200)

# Make prediction on the test set
rf_model_prediction_n200 <- predict(rf_model_n200, newdata = test_data)

# Calculate performance metrics
rf_performance_n200 <- postResample(pred = rf_model_prediction_n200, obs = test_data$SALE_PRC)

# print performance metric
print(rf_performance_n200)
```

```{r}
# Fit model on the training data when n=500
rf_model_n500 <- randomForest(SALE_PRC ~ ., data = train_data, ntree=500)

# Make prediction on the test set
rf_model_prediction_n500 <- predict(rf_model_n500, newdata = test_data)

# Calculate performance metrics
rf_performance_n500 <- postResample(pred = rf_model_prediction_n500, obs = test_data$SALE_PRC)

# print performance metric
print(rf_performance_n500)
```


```{r}
# Create data frame with model names and their corresponding RMSE values
model_names <- c("Linear", "SVR", "Decision Tree", "RF (100)", "RF (200)", "RF (500)")

performance_values <- c(
  lm_performance[1],         
  svr_performance[1],        
  dt_performance[1],         
  rf_performance_n100[1],    
  rf_performance_n200[1],    
  rf_performance_n500[1]
)

# Combines model names with their respective Root Mean Square Error metrics for visualization
performance_df <- data.frame(Model = model_names, RMSE = performance_values)

# Plot using ggplot2
library(ggplot2)
ggplot(performance_df, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model RMSE Comparison", x = "Model", y = "Root Mean Square Error") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```


```{r}
# Define tuning grid for mtry
tune_grid <- expand.grid(.mtry = c(1, 2, 3, 4))

# Cross-validation settings
control <- trainControl(method = "cv", number = 5)

# Train the random forest model with tuning
rf_tuned_model <- train(
  SALE_PRC ~ .,
  data = train_data,
  method = "rf",
  trControl = control,
  tuneGrid = tune_grid,
  ntree = 500  # since this was best previously
)

# Best mtry found
print(rf_tuned_model$bestTune)

# Plot performance vs mtry
plot(rf_tuned_model)
```

```{r}
# Save the fine-tuned model
saveRDS(rf_tuned_model, "rf_tuned_model.rds")
```

```{r}
# Load the fine-tuned model from the saved file
rf_tuned_model <- readRDS("rf_tuned_model.rds")
```

```{r}
# Prepare the new input data replace with the actual data you want to predict
new_data <- data.frame(
  LND_SQFOOT = 11247,
  TOT_LVG_AREA = 4552,
  SPEC_FEAT_VAL = 2105,
  structure_quality = 5
)

# Make prediction on the new data
predicted_price <- predict(rf_tuned_model, newdata = new_data)

# Output the predicted price
print(predicted_price)

```

```{r}

# Check feature importance from the tuned random forest model
importance_values <- varImp(rf_tuned_model)

# View the importance scores
print(importance_values$importance)

# Plot the importance of features
plot(importance_values)
```

