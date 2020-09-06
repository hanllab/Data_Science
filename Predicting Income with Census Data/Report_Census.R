# Project: Predicting Income with Census Data
# Name: Han Lu
# Date: 6/16/2020


########################################
# Install packages and import libraries
########################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

########################################
# Create adult set, train set, test set
########################################

# Census Income Data Set:
# https://archive.ics.uci.edu/ml/datasets/Adult

url_train <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
url_test <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' 
url_names <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names' 

train <- read.table(url_train, sep = ',', stringsAsFactors = FALSE)
test <- readLines(url_test)[-1]
test <- read.table(textConnection(test), sep = ',', stringsAsFactors = FALSE)

names <- readLines(url_names)[97:110]
names <- as.character(lapply(strsplit(names,':'), function(x) x[1])) 
names <- c(names, 'income')
colnames(train) <- names 
colnames(test) <- names

# Remove the missing values (denoted by question marks)

no.question.mark <- apply(train, 1, function(r) !any(r %in% ' ?'))
train <- train[no.question.mark,]
no.question.mark <- apply(test, 1, function(r) !any(r %in% ' ?')) 
test <- test[no.question.mark,]

train <- as.data.frame(unclass(train),stringsAsFactors = T) 
test <- as.data.frame(unclass(test),stringsAsFactors = T)

# Create adult set (combine train, test set)

adult <- rbind(train, test)

# Remove the "." from "<=50K." and ">50K." in adult set

adult$income <- gsub(".", "", as.character(adult$income), fixed = TRUE)


###################
# Data exploration
###################

# ----------------------------
# Explore numerical variables 
# ----------------------------

# Histogram of Age

p1 <- ggplot(train, aes(x = age)) + 
  ggtitle("Histogram of Age") + 
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 5, colour = "black", fill = "#F0E442") + 
  ylab("Percentage") +  
  theme_minimal()

# Histogram of Final Weight

p2 <- ggplot(adult, aes(x = log10(fnlwgt))) +
  ggtitle("Histogram of Final Weight") + 
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  ylab("Percentage") +
  theme_minimal()

# Histogram of Years of Education

p3 <- ggplot(adult, aes(x = education.num)) + 
  ggtitle("Histogram of Years of Education") + 
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 1, colour = "black", fill = "#F0E442") + 
  ylab("Percentage") +
  theme_minimal()

# Histogram of Hours per Week

p4 <- ggplot(adult, aes(x = hours.per.week)) + 
  ggtitle("Histogram of Hours per Week") + 
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  ylab("Percentage") +
  theme_minimal()

# Histogram of Capital Gain

p5 <- ggplot(adult, aes(x = log10(capital.gain+1))) +
  ggtitle("Histogram of Capital Gain") + 
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  ylab("Percentage") +
  theme_minimal()

# Histogram of Capital Loss

p6 <- ggplot(adult, aes(x = log10(capital.loss+1))) + 
  ggtitle("Histogram of Capital Loss") + 
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  ylab("Percentage") +
  theme_minimal()

grid.arrange(p1,p2,p3,p4,p5,p6)

# Percentage of data with zero capital gain

sum(adult$capital.gain==0)/length(adult$capital.gain)*100

# Percentage of data with zero capital loss

sum(adult$capital.loss==0)/length(adult$capital.loss)*100

# -----------------------------
# Explore categorical variables 
# -----------------------------

# Sort categorical variables in descending order

sort.categ <- function(x){reorder(x,x,function(y){length(y)})}
var.categ <- which(sapply(adult, is.factor))
for (c in var.categ){adult[,c] <- sort.categ(adult[,c])}
attach(adult)

# Histogram of Work Class

c1 <- ggplot(adult, aes(y = workclass)) + 
  ggtitle("Histogram of Work Class") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(workclass)) +
  xlab("Percentage") +
  ylab("Work Class") +
  theme_minimal()

# Histogram of Education

c2 <- ggplot(adult, aes(y = education)) + 
  ggtitle("Histogram of Education") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(education)) +
  xlab("Percentage") +
  ylab("Education") +
  theme_minimal()

# Histogram of Marital Status

c3 <- ggplot(adult, aes(y = marital.status)) + 
  ggtitle("Histogram of Marital Status") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(marital.status)) +
  xlab("Percentage") +
  ylab("Marital Status") +
  theme_minimal()

# Histogram of Occupation

c4 <- ggplot(adult, aes(y = occupation)) + 
  ggtitle("Histogram of Occupation") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(occupation)) +
  xlab("Percentage") +
  ylab("Occupation") +
  theme_minimal()

# Histogram of Relationship

c5 <- ggplot(adult, aes(y = relationship)) + 
  ggtitle("Histogram of Relationship") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(relationship)) +
  xlab("Percentage") +
  ylab("Relationship") +
  theme_minimal()

# Histogram of Race

c6 <- ggplot(adult, aes(y = race)) + 
  ggtitle("Histogram of Race") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(race)) +
  xlab("Percentage") + 
  ylab("Race") +
  theme_minimal()

# Histogram of Sex

c7 <- ggplot(adult, aes(y = sex)) + 
  ggtitle("Histogram of Sex") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(sex)) +
  xlab("Percentage") +   
  ylab("Sex") +
  theme_minimal()

# Histogram of Native Country

c8 <- ggplot(adult, aes(y = native.country)) + 
  ggtitle("Histogram of Native Country") + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), colour = "black", fill = "#F0E442") + 
  scale_y_discrete(limits = levels(native.country)) +
  xlab("Percentage") +
  ylab("Native Country") +
  theme_minimal()

grid.arrange(c1,c2,c3,c4,c5,c6)

# -------------------------------------
# Numerical variables and income levels
# -------------------------------------

# Final weight and income

b1 <- ggplot(adult, aes(income, log(fnlwgt))) +
  geom_boxplot(coef=3) +
  xlab("Income") +
  ylab("Final Weight") +
  theme_minimal()
  
# Age and income

b2 <- ggplot(adult, aes(income, age)) +
  geom_boxplot(coef=3) +
  xlab("Income") +
  ylab("Age") +
  theme_minimal()

# Education and income

b3 <- ggplot(adult, aes(income, education.num)) +
  geom_boxplot(coef=3) +
  xlab("Income") +
  ylab("Years of Education") +
  theme_minimal()

# Hours per week and income

b4 <- ggplot(adult, aes(income, hours.per.week)) +
  geom_boxplot(coef=3) +
  xlab("Income") +
  ylab("Hours per Week") +
  theme_minimal()
  
grid.arrange(b1, b2, b3, b4)

# ---------------------------------------
# Categorical variables and income levels
# ---------------------------------------

# Work class and income

ggplot(adult, aes(y = workclass, fill = income)) + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), position='dodge') +
  ggtitle("Work Class and Income") +
  labs(fill = "Income") +
  xlab("Percentage") +
  ylab("Work Class") +
  theme_minimal()

# Occupation and income

ggplot(adult, aes(y = occupation, fill = income)) + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), position='dodge') +
  ggtitle("Occupation and Income") +
  labs(fill = "Income") +
  xlab("Percentage") +
  ylab("Occupation") +
  theme_minimal()

# Education and income

ggplot(adult, aes(y = education, fill = income)) + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), position='dodge') +
  ggtitle("Education and Income") +
  labs(fill = "Income") +
  xlab("Percentage") +
  ylab("Education") +
  theme_minimal()

# Marital status and income

ggplot(adult, aes(y = marital.status, fill = income)) + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), position='dodge') +
  ggtitle("Marital Status and Income") +
  labs(fill = "Income") +
  xlab("Percentage") +
  ylab("Marital Status") +
  theme_minimal()

# Relationship and income

ggplot(adult, aes(y = relationship, fill = income)) + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), position='dodge') +
  ggtitle("Relationship and Income") +
  labs(fill = "Income") +
  xlab("Percentage") +
  ylab("Relationship") +
  theme_minimal()

# Race and income

ggplot(adult, aes(y = race, fill = income)) + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), position='dodge') +
  ggtitle("Race and Income") +
  labs(fill = "Income") +
  xlab("Percentage") +
  ylab("Race") +
  theme_minimal()

# Sex and income

ggplot(adult, aes(y = sex, fill = income)) + 
  geom_bar(aes(x = 100*(..count..)/sum(..count..)), position='dodge') +
  ggtitle("Sex and Income") +
  labs(fill = "Income") +
  xlab("Percentage") +
  ylab("Sex") +
  theme_minimal()


######################
# Supervised Learning
######################

# --------------
# Data cleaning 
# --------------


# Combine capital.gain and capital.loss into a single capital.change variable in train and test set

train$capital.change <- train$capital.gain - train$capital.loss
test$capital.change <- test$capital.gain - test$capital.loss

train$capital.gain <- NULL
train$capital.loss <- NULL
test$capital.gain <- NULL
test$capital.loss<-NULL

# Switch income and capital.change columns (let income be the last column)

train[c(11,12)] <- train[c(12,11)]
colnames(train)[11:12] <- colnames(train)[12:11]
test[c(11,12)] <- test[c(12,11)]
colnames(test)[11:12] <- colnames(test)[12:11]


# Delete education variable in train and test set

train$education <- NULL
test$education <- NULL

# Delete native.country variable in train and test set

train$native.country <- NULL
test$native.country <- NULL

# Convert income to dummy variable

train$income <- as.factor(ifelse(train$income == ' <=50K', 0, 1))
test$income <- as.factor(ifelse(test$income == ' <=50K.', 0, 1))

# --------------------
# Classification Tree 
# --------------------

set.seed(1, sample.kind = "Rounding")
tree <- rpart(income ~ ., data = train, method = 'class')
tree.hat <- predict(tree, newdata = test, type = 'class')
confusionMatrix(tree.hat, test$income)$overall["Accuracy"]

# ---------------
# Random Forrest
# ---------------

set.seed(1, sample.kind="Rounding")
forest <- randomForest(train$income ~ ., data = train, mtry = sqrt(10), importance = TRUE) 
forest

forest.hat <- predict(forest, newdata = test, type = "class") 
confusionMatrix(forest.hat, test$income)$overall["Accuracy"]

varImpPlot(forest, type = 2, main = "Variable Importance Plot")


###########
# The End
###########