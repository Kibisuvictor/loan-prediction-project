library(tidymodels)
library(tidyverse)
library(janitor)
library(DataExplorer)
library(skimr)
library(naniar)

#import the dataset
train <- read_csv("C:\\Users\\hope\\Downloads\\data science projects/train_loan.csv")
test <- read_csv("C:\\Users\\hope\\Downloads\\data science projects/test_loan.csv")
train <- train %>% clean_names()
test <- test %>% clean_names()
train %>% head()
test %>% head()
dim(train)
dim(test)

#explore the data
train %>% plot_bar(maxcat = 50, ncol = 3)
train %>% plot_histogram()
train %>% skim()
train %>% str()

#changing of some key features
train$credit_history <- as.factor(train$credit_history)
test$credit_history <- as.factor(test$credit_history)

train <- train %>% mutate_if(is.character, as.factor)
test <- test %>% mutate_if(is.character,as.factor)
#check the missing values
train %>% miss_var_summary()
train %>% skim(loan_amount_term)
train$dependents %>% table()
test$loan_amount_term %>% table()
train$loan_status %>% table()

#split our data
train_split <- train %>% initial_split(prop = .7, strata = loan_status)
trained <- training(train_split)

#folds for cross validation
folds <- vfold_cv(trained, v = 10)

#prepare a recipe
rec_prep <- recipe(loan_status ~., data = train) %>% 
  step_rm(loan_id) %>% 
  step_nzv(all_predictors()) %>% 
  step_knnimpute(all_predictors()) %>% 
  step_YeoJohnson(all_numeric()) %>% 
  step_corr(all_numeric()) %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric()) %>% 
    step_dummy(all_nominal(), -all_outcomes())

#lets fit a random forest model
rf_mod <- rand_forest(mode = "classification",
                      mtry = tune(),
                      trees = tune(),
                      min_n = tune()) %>% 
  set_engine("ranger")

#workflow
rf_wkflow <- workflow() %>% 
  add_recipe(rec_prep) %>% 
  add_model(rf_mod)

#hyperparametrs
rf_grid <- grid_random(mtry() %>% range_set(c(2,7)),
                       min_n(),
                       trees(),
                       size = 20)
#tuning the parameters
rf_tune <- tune_grid(rf_wkflow,
                     resamples = folds,
                     grid = rf_grid,
                     metrics = metric_set(accuracy),
                     control = control_grid(save_pred = TRUE))
rf_tune %>% collect_metrics()
rf_tune %>% show_best()
rf_params <- rf_tune %>% select_best()
raf_workflow <- rf_wkflow %>% finalize_workflow(rf_params)
rf_fit <- raf_workflow %>% last_fit(train_split)
rf_fit %>% collect_metrics()

rf_final <- raf_workflow %>% fit(train)
rf_preds <- rf_final %>% predict(test) %>% as_vector()
rf_preds

rf_pred <- tibble(
  Loan_ID = test$loan_id,
  Loan_Status = rf_preds
)
write_csv(rf_pred, "raf2.csv")
