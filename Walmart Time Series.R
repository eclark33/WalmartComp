# libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(prophet)



# read in data
train_data <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/WalmartComp/train.csv")
testData <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/WalmartComp/test.csv")

# additional data sets
stores <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/WalmartComp/stores.csv")
features <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/WalmartComp/features.csv")


############################# DATA PREPPING ###############################

# create day of week variable
train_data <- train_data %>%
  mutate(date_doy = yday(Date))


# replace markdown NAs with 0s
features <- features %>% 
  mutate(MarkDown1 = replace_na(MarkDown1, 0),
         MarkDown2 = replace_na(MarkDown2, 0),
         MarkDown3 = replace_na(MarkDown3, 0),
         MarkDown4 = replace_na(MarkDown4, 0),
         MarkDown5 = replace_na(MarkDown5, 0))

# create TotalMarkdown variable (sum of markdowns)
features <- features %>%
  mutate(TotalMarkdown = rowSums(across(c(MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5))))

# create MarkdownFlag variable (1 if markdown 0 if not)
features <- features %>%
  mutate(MarkdownFlag = if_else(TotalMarkdown != 0, 1, 0))

# impute Missing CPI and unemployment
feature_recipe <- recipe(~., data=features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment,
                  impute_with = imp_vars(DecDate, Store))

imputed_features <- juice(prep(feature_recipe))

# join
train_data <- train_data %>%
  left_join(imputed_features, by = c("Store", "Date"))

## JOIN WITH TEST DATA ##
# create day of week variable
testData <- testData %>%
  mutate(date_doy = yday(Date))

testData <- testData %>%
  left_join(imputed_features, by = c("Store", "Date"))


############################# EDA ###############################
# look at stores & departments in test and training sets
stores_test <- unique(testData$Store)
stores_train <- unique(train_data$Store)

stores_test
stores_train

# look at sample size for each store and department 
sample_sizes <- train_data %>% group_by(Store, Dept) %>%
  summarize(n = n())

# look at holidays
train_data %>%
  filter(IsHoliday.x == TRUE) %>%    
  mutate(Month = month(Date, label = TRUE)) %>%   
  group_by(Month) %>%
  summarise(n = n(), .groups = "drop") %>%
  arrange(Month)


# time series plot
train_data %>% 
  filter(Store == 1, Dept == 1) %>%
  ggplot(aes(x = Date, y = Weekly_Sales)) +
  geom_line() +
  geom_smooth(se = FALSE)


################################# MODELING #######################################

all_preds <- tibble(Id = character(), Weekly_Sales = numeric())

# Get all unique store–dept combos in test set
combos <- testData %>% distinct(Store, Dept)

for(i in seq_len(nrow(combos))) {
  
  store <- combos$Store[i]
  dept  <- combos$Dept[i]
  
  # Filter train/test for this combo
  dept_train <- train_data %>% filter(Store == store, Dept == dept)
  dept_test  <- testData  %>% filter(Store == store, Dept == dept)
  
  # Handle edge cases
  if(nrow(dept_train) == 0) {
    preds <- dept_test %>%
      transmute(Id = paste(Store, Dept, Date, sep = "_"),
                Weekly_Sales = 0)
    
  } else if(nrow(dept_train) < 10) {
    preds <- dept_test %>%
      transmute(Id = paste(Store, Dept, Date, sep = "_"),
                Weekly_Sales = mean(dept_train$Weekly_Sales))
    
  } else {
    # Recipe
    my_recipe <- recipe(Weekly_Sales ~ ., data = dept_train) %>%
      step_range(date_doy, min = 0, max = pi) %>%
      step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy)) %>%
      step_date(Date, features = c("month","year")) %>%
      step_rm(any_of(c("Date", "Store", "Dept", "IsHoliday", "IsHoliday.x", "IsHoliday.y")))
    
    
    # Model
    forest_mod <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 100) %>%
      set_engine("ranger") %>%
      set_mode("regression")
    
    wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(forest_mod)
    
    # CV folds (time‑series CV would be even better)
    folds <- vfold_cv(dept_train, v = 3)
    
    grid <- grid_regular(mtry(range = c(1,10)),
                         min_n(),
                         levels = 5)
    
    tuned <- tune_grid(
      wf,
      resamples = folds,
      grid = grid,
      metrics = metric_set(mae)
    )
    
    bestTune <- tuned %>% select_best(metric = "mae")
    
    final_wf <- wf %>%
      finalize_workflow(bestTune) %>%
      fit(data = dept_train)
    
    preds <- dept_test %>%
      transmute(Id = paste(Store, Dept, Date, sep = "_"),
                Weekly_Sales = predict(final_wf, new_data = .) %>% pull(.pred))
  }
  
  # Bind predictions
  all_preds <- bind_rows(all_preds, preds)
  
  cat("Store", store, "Dept", dept, "done.\n")
}

# Save predictions
vroom::vroom_write(all_preds, "Walmart_RF_Preds.csv", delim = ",")






























# recipe
walmart_recipe <- recipe(Weekly_Sales ~ ., data = train_data) %>%
                    step_range(date_doy, min = 0, max = pi) %>%
                      step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))
                     





# find all combinations 
store_dept_pairs <- train_data %>%
  distinct(Store, Dept)

sampled_pairs <- store_dept_pairs %>%
  sample_n(1)



################ store 1 dept 80 ##############
s1dept80 <- train_data %>%
  filter(Dept == 80 & Store == 1)

# model using random forest
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


# forest recipe
s1dept80_recipe <- recipe(Weekly_Sales ~ ., data = s1dept80) %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))


# workflow  
s1dept80_workflow <- workflow() %>%
  add_recipe(s1dept80_recipe) %>%
  add_model(forest_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(mtry(range = c(1,10)),
                                      min_n(),
                                      levels = 3) 

# split data for cv & run it 
folds <- vfold_cv(s1dept80, v = 5, repeats = 1)

CV_results <- s1dept80_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mae), 
            control = control_grid(verbose = TRUE))


# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "mae")

## finalize the workflow & fit 
final_wf <- s1dept80_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = s1dept80) 


## predictions
boost_preds <- final_wf %>%
  predict(new_data = testData)










##################### store 22 dept 35 ####################
s22dept35 <- train_data %>%
  filter(Dept == 22 & Store == 35)

# model using random forest
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

s22dept35 <- s22dept35 %>%
  rename(IsHoliday = IsHoliday.x) %>%
  select(-IsHoliday.y)


# forest recipe
s22dept35_recipe <- recipe(Weekly_Sales ~ ., data = s22dept35) %>%
  step_mutate(Store = factor(Store),
              Dept = factor(Dept),
              IsHoliday = factor(IsHoliday)) %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy)) %>%
  step_normalize(all_numeric_predictors()) 


# workflow  
s22dept35_workflow <- workflow() %>%
  add_recipe(s22dept35_recipe) %>%
  add_model(forest_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(mtry(range = c(1,10)),
                                      min_n(),
                                      levels = 3) 

# split data for cv & run it 
folds <- vfold_cv(s22dept35, v = 5, repeats = 1)

CV_results <- s22dept35_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mae), 
            control = control_grid(verbose = TRUE))


# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "mae")

## finalize the workflow & fit 
final_wf <- s22dept35_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = s22dept35) 


show_best(CV_results, metric = "mae", n = 1)





###################### store 15 dept 50 ###################
s15dept50 <- train_data %>%
  filter(Dept == 50 & Store == 15)


# model using random forest
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

s15dept50 <- s15dept50 %>%
  rename(IsHoliday = IsHoliday.x) %>%
  select(-IsHoliday.y)


# forest recipe
s15dept50_recipe <- recipe(Weekly_Sales ~ ., data = s15dept50) %>%
  step_mutate(Store = factor(Store),
              Dept = factor(Dept),
              IsHoliday = factor(IsHoliday)) %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy)) %>%
  step_mutate(Holiday_Name = case_when(
      Date %in% as.Date(c("2010-02-12","2011-02-11","2012-02-10","2013-02-08")) ~ "SuperBowl",
      Date %in% as.Date(c("2010-09-10","2011-09-09","2012-09-07","2013-09-06")) ~ "LaborDay",
      Date %in% as.Date(c("2010-11-26","2011-11-25","2012-11-23","2013-11-29")) ~ "Thanksgiving",
      Date %in% as.Date(c("2010-12-31","2011-12-30","2012-12-28","2013-12-27")) ~ "Christmas",
      TRUE ~ "None")) %>%
  step_mutate(Holiday_Name = factor(Holiday_Name)) %>%
  step_lag(Weekly_Sales, lag = c(1, 2, 3, 52)) %>%
  step_impute_mean(starts_with("Weekly_Sales_lag"))



# workflow  
s15dept50_workflow <- workflow() %>%
  add_recipe(s15dept50_recipe) %>%
  add_model(forest_mod)

# grid of values to tune 
grid_of_tuning_params <- grid_regular(mtry(range = c(1,10)),
                                      min_n(),
                                      levels = 3) 

# split data for cv & run it 
folds <- vfold_cv(s15dept50, v = 5, repeats = 1)

CV_results <- s15dept50_workflow %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(mae), 
            control = control_grid(verbose = TRUE))


# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "mae")

## finalize the workflow & fit 
final_wf <- s15dept50_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = s15dept50) 


show_best(CV_results, metric = "mae", n = 1)




######################### PROPHET MODEL #########################


# Store and Dept
store <- 4 # I did 17
dept <- 34 # I used 17
  
# filter and Rename to match prophet syntax
sd_train <- train_data %>%
  filter(Store == store, Dept == dept) %>%
  rename(y = Weekly_Sales, ds = Date)

sd_test <- testData %>%
  filter(Store == store, Dept == dept) %>%
  rename(ds = Date)


sd_test <- sd_test %>%
  fill(CPI, .direction = "downup")
sd_train <- sd_train %>%
  fill(Unemployment, .direction = "downup")
sd_test <- sd_test %>%
  fill(Unemployment, .direction = "downup")


# fit prophet model
prophet_model <- prophet() %>%
  add_regressor("date_doy") %>%
  add_regressor("Fuel_Price") %>%
  add_regressor("CPI") %>%
  add_regressor("Unemployment") %>%
  add_regressor("TotalMarkdown") %>%
  fit.prophet(df = sd_train)


## Predict Using Fitted prophet Model
fitted_vals <- predict(prophet_model, df = sd_train) 

#For Plotting Fitted Values
test_preds <- predict(prophet_model, df = sd_test) #Predictions are called "yhat"

## Plot Fitted and Forecast on Same Plot
plot_uno <- ggplot() + 
  geom_line(data = sd_train, mapping = aes(x = ds, y = y, color = "Data")) +
  geom_line(data = fitted_vals, mapping = aes(x = as.Date(ds), y = yhat, color = "Fitted")) +
  geom_line(data = test_preds, mapping = aes(x = as.Date(ds), y = yhat, color = "Forecast")) +
  scale_color_manual(values = c("Data" = "black", "Fitted" = "blue", "Forecast" = "red")) + 
  labs(color = "")

plot_dos <- ggplot() + 
  geom_line(data = sd_train, mapping = aes(x = ds, y = y, color = "Data")) +
  geom_line(data = fitted_vals, mapping = aes(x = as.Date(ds), y = yhat, color = "Fitted")) +
  geom_line(data = test_preds, mapping = aes(x = as.Date(ds), y = yhat, color = "Forecast")) +
  scale_color_manual(values = c("Data" = "black", "Fitted" = "blue", "Forecast" = "red")) + 
  labs(color = "")



library(patchwork)

combined_plot <- plot_uno / plot_dos   
combined_plot

