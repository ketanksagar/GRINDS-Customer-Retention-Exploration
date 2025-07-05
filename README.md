# GRINDS-Customer-Retention-Exploration
As part of our short-term consulting immersion week, my team and I worked with the Indiana company, Grinds. We reviewed their data and built models to analyze market trends to develop insights regarding analytics, growth strategies, and market opportunities. We wanted to build a model that predicted the customers likely to return so the company could use it to then target those customers with specific marketing strategies tailored to increase repeat customers, sales, and average order values. 


## Importing the relevant packages needed:
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)

library(ggplot2)
library(ggthemes)
library("randomForest")
library("rpart")
library("caret")
library("splitstackshape")
library("DMwR")
library("nnet")
library("xgboost")
library("gam")

library(stringr)
```

## Data Preparation

```{r}
# Read the CSV file into a data frame
returning_data1 = read.csv("C:\\Users\\Graduate\\Downloads\\sales_2023-12-01_2024-11-01_Grinds Confidential.csv", header = TRUE)

# Filter out rows where net_quantity is less than 0, keeping only those with zero (indicating returns)
returning_data1 <- returning_data1[returning_data1$net_quantity >= 0, ]

# Cleaning column names and removing the special attributes
colnames(returning_data1) <- make.names(colnames(returning_data1))

# Convert the product_title column to UTF-8 encoding, removing invalid characters
Encoding(returning_data1$product_title) <- "UTF-8"

returning_data1$product_title <- iconv(returning_data1$product_title, 
                                      "UTF-8", "UTF-8", sub = "")

returning_data1$product_title <- gsub("^\\W\\s", "", returning_data1$product_title, perl = TRUE)


# Replace billing_region names not found in the list of US states with 'International'
returning_data1$billing_region <- ifelse(!(returning_data1$billing_region %in% state.name), "International", returning_data1$billing_region)

# Remove blanks in product_vendor
returning_data1 <- returning_data1[returning_data1$product_vendor != "", ]

# Convert all product titles to lowercase to standardize the data
returning_data1$product_title <- tolower(returning_data1$product_title)

# Rename specific product titles for consistency
returning_data1$product_title[returning_data1$product_title == 'grinds coffee pouches - vanilla'] <- 'vanilla'
returning_data1$product_title[returning_data1$product_title == 'grinds coffee pouches - 3pack variety with double mocha, sweet mint and cinnamon roll'] <- '3 pack variety'
returning_data1$product_title[returning_data1$product_title == '3-pack energy sampler'] <- '3 pack energy sampler'
returning_data1$product_title[returning_data1$product_title == 'Mocha'] <- 'DoubleMocha'

# Remove dashes
returning_data1$product_title <- gsub("-.*", "", returning_data1$product_title)

# Convert product titles to title case for better readability
returning_data1$product_title <- str_to_title(returning_data1$product_title)

# Convert product titles to title case for better readability
returning_data1$product_title <- gsub(" ", "", returning_data1$product_title)

# Remove '100% off'
returning_data1$product_title <- gsub("\\(100%off\\)", "", returning_data1$product_title, ignore.case = TRUE)

# Display the first few rows of the cleaned data frame
head(returning_data1)

# Pull unique titles
unique(returning_data1$product_title)
```
## Data Partitioning 

Doing stratified to ensure there's a more balanced separation between response variables in both training and test sets.
```{r}
library(dplyr)
returning_data <- returning_data1 %>% 
  select(-customer_id, -billing_city, -product_vendor) %>%  # Remove variables that are not necessary for the model
  mutate(customer_type = ifelse(customer_type == "First-time", 0, 1)) # Convert customer_type to binary format (0: First-time, 1: Returning)
          
# Split the data into training and testing sets while maintaining the proportion of the customer_type variable
split_data <- stratified(returning_data, 
                        group = c("customer_type"), 
                        size = 0.8, bothSets = TRUE)
```

Partitioning data to training and test sets: 
```{r}
# removing any rows with NA values
train_data <- as.data.frame(na.omit(split_data[[1]]))

test_data <- na.omit(split_data[[2]])
```


## Logsitic Regression

Fit a logistic regression model to predict customer_type
```{r}
fit_1 <- glm(customer_type ~ discounts+returns+billing_region, # Set formula
             family=binomial, # Set logistic regression
             data= train_data) # Set data set
t1 <- summary(fit_1) # Summarize model
t1
```

Analyzing the coefficients to see which are significant (greater tha 0.01):

```{r}
t1$coefficients[which(t1$coefficients[,4] >= 0.01),]
```

## Model Evaluation

Use the test data for predictions and evaluate the model's performance

```{r}
library(caret)

fit_1_pred <- predict(fit_1, newdata=test_data, type='response')
```

Evaluating the predictive performance:

```{r Q5(c)}
lm_full_acc <- confusionMatrix(factor(ifelse(fit_1_pred>0.5, '1', '0')), as.factor(test_data$customer_type), positive='1')
```

```{r}
# To simplify our comparison, get the results in an easier format:
print(c(lm_full_acc$overall[1], lm_full_acc$byClass[1:2]))
```

The logistic regression doesn't provide great accuracy, having only a 58.5% score. Many variables couldn't be included as well since their coefficient values were so high that they were not allowing the logistic regression to create a sigmoid function and run it's model.

Let's do another model:

## GAM

Convert some variables to factors
```{r}
returning_data1$product_title <- as.factor(returning_data1$product_title)
returning_data1$customer_type <- as.factor(returning_data1$customer_type)
returning_data1$billing_region <- as.factor(returning_data1$billing_region)
```

Adjusting some product titles:
```{r}
# Rename
returning_data1$product_title[returning_data1$product_title == 'mocha'] <- 'doublemocha'
returning_data1$product_title[returning_data1$product_title == 'cherry'] <- 'peach' # putting it with another fruity flavor since there isn't any other to substitute it with
```


Sample data and removing NA values:
```{r}
split_data1 <- stratified(returning_data1, 
                        group = c("customer_type"), 
                        size = 0.8, bothSets = TRUE)

train <- as.data.frame(na.omit(split_data1[[1]]))

test <- as.data.frame(na.omit(split_data1[[2]]))
```

Adjusting some predictors and completing the model:
```{r}
# Taxes and discounts as percent of gross_sales
train$taxes_per <- train$taxes/train$gross_sales
test$taxes_per <- test$taxes/test$gross_sales
train$discounts_per <- abs(train$discounts/train$gross_sales)
    train$discounts_per <- as.numeric(gsub("NaN", "100", train$discounts_per))
    train$discounts_per <- as.numeric(gsub("Inf", "0", train$discounts_per))
test$discounts_per <- abs(test$discounts/test$gross_sales)
    test$discounts_per <- as.numeric(gsub("NaN", "100", test$discounts_per))
    test$discounts_per <- as.numeric(gsub("Inf", "0", test$discounts_per))

# Build on train data
gam1 <- gam(customer_type~product_title+billing_region+s(net_quantity)+s(discounts_per)+s(returns)+s(total_sales), family = 'binomial', data = train)

# Get the summary of the model
summary(gam1)


#plot(gam1, col='blue')
```
# Evaluation on test data
```{r}
gam1_pred <- predict(gam1, newdata=test, type='response')
gam1_acc <- confusionMatrix(factor(ifelse(gam1_pred>0.5, 'Returning', 'First-time')), test$customer_type, positive='Returning')
print(gam1_acc)
```
The GAM model is showing us a much better accuracy score of 65%, doing a much better job at predicting the customers more likely to return.



## Developing First Tree
```{r}
# Making our first tree based on the training data
Tree_1 <- randomForest(customer_type ~ . -product_title -customer_id, 
                              data = train_data, 
                              mtry = ncol(train_data) * .333,
                              ntree = 150,
                              nodesize = 750)

# Testing our tree
test_1 <- predict(Tree_1, newdata = test_data, type = 'response')

# Comparing it to the original results
tree_1_acc <- confusionMatrix(test_1, test_data$customer_type, positive = 'Returning')

# Getting the important factors from the model
tree_importance <- as.data.frame(Tree_1$importance)

# Writing them out so they can be graphed
tree_importance$Variable <- rownames(tree_importance)

# Visualizing the plot
ggplot(tree_importance, aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "red") +
  coord_flip() +
  labs(title = "Variable Importance (Random Forest)",
       x = "Variables",
       y = "Mean Decrease in Gini") +
  theme_minimal()
```
The main takeaway from this first tree was to get rid of the taxes and the returns. We also knew we needed to add in the flavors down the road.

## Building the 2nd Tree
```{r}
# Creating a Random Forest model to predict customer type
Tree_2 <- randomForest(customer_type ~ . 
                       -product_title -customer_id -taxes -returns, 
                       data = train_data, 
                       mtry = sqrt(ncol(train_data)), # Number of variables to possibly split at each node
                       ntree = 100,                     # Number of trees to grow
                       nodesize = 500)                  # Minimum size of terminal nodes

# Testing the trained Random Forest model on new data
test_2 <- predict(Tree_2, newdata = test_data, type = 'response')

# Evaluating model performance using a confusion matrix
tree_2_acc <- confusionMatrix(test_2, test_data$customer_type, positive = 'Returning')

# Extracting the variable importance of the model
tree_2_importance <- as.data.frame(Tree_2$importance)
tree_2_importance$Variable <- rownames(tree_2_importance) # Creating a variable for better visualization

# Visualizing variable importance using a horizontal bar plot
ggplot(tree_2_importance, aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "red") +        # Bar chart with red fill
  coord_flip() +                                   # Flipping the coordinates for horizontal bars
  labs(title = "Variable Importance (Random Forest)", # Adding titles and labels
       x = "Variables",
       y = "Mean Decrease in Gini") +
  theme_minimal()                                   # Minimal theme for aesthetics
```
Now that taxes and returns were gone, we felt that everything was fairly significant and the model was as simple as it could reasonably be. Now, to add in the flavors. 

The main issue with the flavors was the quantity. There were quite a few when we started, and even when we cleaned the data, it was still above the amount that the random forest felt comfortable using as a factor. 

So we had to get rid of a few more products.

## Creating a new set:
```{r}
# Making a new set
returning_data_2 <- returning_data

# Dropping a combination of temporary flavors, discontinued flavors, and non-flavor based products to reduce the number of products listed
returning_data_2 <- returning_data_2[
                     returning_data_2$product_title != "SteakSauce" &
                     returning_data_2$product_title != "NewYear'sResolutionsPack" &
                     returning_data_2$product_title != "12OzGrindsTumbler" &
                     returning_data_2$product_title != "GrindsHandcraftedMug" &
                     returning_data_2$product_title != "GrindsPintGlass" &
                     returning_data_2$product_title != "GrindsFlask" &
                     returning_data_2$product_title != "BlackFridayTop", ]

# Converting the customer type into 1 and 0s so we can perform regression and return a %
returning_data_2$customer_type <- ifelse(returning_data_2$customer_type == 'Returning', 1, 0)

# Converting products to factors
returning_data_2$product_title <- as.factor(returning_data_2$product_title)
```
```{r}
# Splitting the dataset based on customer type, stratifying by group
split_data_2 <- stratified(returning_data_2, 
                        group = c("customer_type"),     # Stratification by customer type
                        size = 0.8,                     # Training set size as 80%
                        bothSets = TRUE)                # Return both training and test sets

# Creating the training data set, removing any NA values
train_data_2 <- as.data.frame(na.omit(split_data_2[[1]]))

# Creating the testing data set, removing any NA values
test_data_2 <- na.omit(split_data_2[[2]])

# Converting customer_type variable into a factor for modeling
train_data_2$customer_type <- as.factor(train_data_2$customer_type)
```

The product based tree:
```{r}
Product_Tree <- randomForest(customer_type ~ . -customer_id -taxes -returns, 
                              data = train_data_2, 
                              mtry = sqrt(ncol(train_data_2)), # Number of variables to possibly split at each node
                              ntree = 150,                       # Number of trees to grow
                              nodesize = 500)                    # Minimum size of terminal nodes

# Using the product-based model to predict customer type on test data
test_3 <- predict(Product_Tree, newdata = test_data_2)

# Transforming predictions to numeric for further processing
test_3n <- as.numeric(test_3) - 1

# Scaling into 1 and 0 for accurate grading
pred_3_fact <- as.factor(ifelse(test_3n >= 0.5, "1", "0")) 
```

Evaluating and visualizing the model:
```{r}
# Evaluating model performance using a confusion matrix
tree_3_acc <- confusionMatrix(pred_3_fact, as.factor(test_data_2$customer_type), positive = '1')

# Plotting the predicted probabilities to visualize customer return likelihood
customer_dist <- ggplot(data = data.frame(predictions = test_3n), 
                        aes(x = predictions)) +
  geom_histogram(binwidth = 0.05, fill = "lightblue") + # Histogram for predicted probabilities
  labs(title = "Distribution of Customer Chance of Return", 
       x = "Predicted Probability", y = "Count") +       # Adding titles and labels
  theme_minimal(base_size = 15) +                        # Minimal theme for aesthetics
  theme(plot.background = element_rect(fill = "transparent", color = NA), # Transparent plot background
        panel.background = element_rect(fill = "transparent"))

# Visualizing the distribution of customer return probability
customer_dist
```


We then decided to test how likely the average customer was to be returning when we factored out the returning customers, as those who had a high volume order and ordered more specific flavors were more likely to become returning customers.

```{r}
# Creating a new test set containing only customers classified as '0'
test_data_First <- returning_data_2[returning_data_2$customer_type == '0', ]

# Making the predictions
test_first <- predict(Product_Tree, newdata = test_data_First)

# Generating the plot
customer_dist_first <- ggplot(data = data.frame(predictions = test_first), 
                              aes(x = predictions)) +
                          geom_histogram(binwidth = 0.05, fill = "lightblue") +
                          labs(title = "Distribution of Customer Chance of Return", 
                               x =  "Predicted Probability", y = "Count") + 
                          theme_minimal(base_size = 15) + 
                          theme(plot.background = element_rect(fill = 'NA', color = NA),
                                panel.background = element_rect(fill = 'NA'))

# Writing the plot out for later usage
png("histogram_transparent_first.png", bg = "transparent")


hist(as.numeric(test_first)-1, 
     main = "Distribution of Customer Chance of Return", 
     xlab = "Predicted Probability", 
     col = "lightblue", 
     border = "black")

dev.off()
```

Like we predicted, the odds the average customer would become a returning customer dropped dramatically when we removed customers who we knew became returning customers.

```{r}
summary(test_first)
```
Compared to 

```{r}
summary(test_3)
```

In summary, the average customer became about 7% less likely to become a returning customer.

