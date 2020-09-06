# Project: MovieLens Project Report - Building a Movie Recommendation System
# Name: Han Lu
# Date: 6/6/2020

################################
# Create edx set, validation set
################################

# Import packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


################ Split edx into Train, Test ###################

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set

test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into train set

removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(test_index, temp, removed)

####################### Data Exploration ######################

# Overview of edx

glimpse(edx)
summary(edx)

# Summarize date

tibble('Start Date' = date(as_datetime(min(edx$timestamp), origin="1970-01-01")),
       'End Date' = date(as_datetime(max(edx$timestamp), origin="1970-01-01"))) %>%
  mutate(Span = duration(max(edx$timestamp)-min(edx$timestamp)))

# Number of users in the edx set

length(unique(edx$userId))

# Number of movies in the edx set

length(unique(edx$movieId))

# Graph - Distribution of movies

edx %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Movies by the Number of Ratings", 
          subtitle = "Some movies get rated more than others.") +
  xlab("Number of Ratings") +
  ylab("Number of Movies") + 
  theme_economist()

# Graph - Distribution of users

edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Users by the Number of Ratings", 
          subtitle="Some users are more active than others.") +
  xlab("Number of Ratings") +
  ylab("Number of Users") + 
  scale_y_continuous(labels = comma) + 
  theme_economist()

# User-movie sparse matrix

users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
title("User-Movie Matrix")

# Most rated movies

edx %>% group_by(title) %>%
  summarize(n= n()) %>%
  arrange(desc(n))

# Number of movies rated once

edx %>% group_by(title) %>%
  summarize(n = n()) %>%
  filter(n==1) %>%
  count() %>%
  pull()

# Count the number of each rating

edx %>% group_by(rating) %>% summarize(n=n())

length(unique(edx$genres))

# Genres: 797 combos

length(unique(edx$genres))

#################### Define Loss Functions #################

# Define Mean Absolute Error (MAE)

MAE <- function(true_ratings, predicted_ratings){
  mean(abs(true_ratings - predicted_ratings))
}

# Define Mean Squared Error (MSE)

MSE <- function(true_ratings, predicted_ratings){
  mean((true_ratings - predicted_ratings)^2)
}

# Define Root Mean Squared Error (RMSE)

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

######################### Mean MODEL #######################

# Mean of the observed values in the training set

mu <- mean(train$rating)

# Report results

result <- bind_rows(tibble(Method = "Mean", 
                           RMSE = RMSE(test$rating, mu),
                           MSE  = MSE(test$rating, mu),
                           MAE  = MAE(test$rating, mu)))
result

#################### Movie Effect Model ####################

# Create movie effect term (b_i)

b_i <- train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
head(b_i)

# Predict ratings with mu and b_i

y_hat <- test %>%
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Report results

result <- bind_rows(result, 
                    tibble(Method = "Movie Effect", 
                           RMSE = RMSE(test$rating, y_hat),
                           MSE  = MSE(test$rating, y_hat),
                           MAE  = MAE(test$rating, y_hat)))
result

################ Movie and User Effect Model ################

# Create user effect term (b_u)

b_u <- train %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings with mu, b_i, and b_u

y_hat <- test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Update results

result <- bind_rows(result, 
                    tibble(Method = "Movie and User Effect", 
                           RMSE = RMSE(test$rating, y_hat),
                           MSE  = MSE(test$rating, y_hat),
                           MAE  = MAE(test$rating, y_hat)))
result

###################### Regularization #######################

# Define lambdas

lambdas <- seq(from=0, to=10, by=0.25)

# Compute RMSE for each lambda

rmses <- sapply(lambdas, function(l){
  
  # Mean rating
  mu <- mean(train$rating)
  
  # Movie effect (b_i)
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # User effect (b_u)
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Predictions from y_hat = mu + b_i + b_u
  predicted_ratings <- test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  # Report RMSE
  return(RMSE(predicted_ratings, test$rating))
})

# Plot the lambda vs RMSE

tibble(Lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  ggtitle("Regularization", 
          subtitle = "The optimal penalization gives the lowest RMSE.") +
  theme_economist()

# Define the optimal lambda

lambda <- lambdas[which.min(rmses)]

##################### Regularizaed Model ####################

# Regularized movie effect (b_i)

b_i <- train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# Regularized user effect (b_u)

b_u <- train %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Prediction

y_hat <- test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Report results
result <- bind_rows(result, 
                    tibble(Method = "Regularized Movie and User Effect", 
                           RMSE = RMSE(test$rating, y_hat),
                           MSE  = MSE(test$rating, y_hat),
                           MAE  = MAE(test$rating, y_hat)))
result

###################### Final Validation #####################

# Mean rating

mu <- mean(edx$rating)

# Regularized movie effect (b_i)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# Regularized user effect (b_u)

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Prediction

y_hat <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Report results

result <- bind_rows(result, 
                    tibble(Method = "Final Validation", 
                           RMSE = RMSE(validation$rating, y_hat),
                           MSE  = MSE(validation$rating, y_hat),
                           MAE  = MAE(validation$rating, y_hat)))
result

###################### Some Predictions #####################

# Top 10 best movies

validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  arrange(-pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)

# Top 10 worst movies

validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  arrange(pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)


###########
# The End
###########