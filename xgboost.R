?boost_tree
xg_mod <- boost_tree(mode = "classification",
                     mtry = tune(),
                     trees = tune(),
                     min_n = tune(),
                     learn_rate = tune()) %>% 
  set_engine("xgboost")

xg_wkflw <- workflow() %>% 
  add_recipe(rec_prep) %>% 
  add_model(xg_mod)

xg_grid <- grid_random(mtry() %>% range_set(c(2,7)),
                       trees(),
                       min_n(),
                       learn_rate(),
                       size = 20)

xg_tune <- tune_grid(xg_wkflw,
                     resamples = folds,
                     grid = xg_grid,
                     metrics = metric_set(accuracy),
                     control = control_grid(save_pred = TRUE))
xg_tune %>% collect_metrics()
xg_tune %>% show_best()
xg_params <- xg_tune %>% select_best()
xg_workflow <- xg_wkflw %>% finalize_workflow(xg_params)
xg_fit <- xg_workflow %>% last_fit(train_split)
xg_fit %>% collect_metrics()
xg_final <- xg_workflow %>% fit(train)

xg_preds <- xg_final %>% predict(test) %>% as_vector()
xg_preds
xg_pred <- tibble(
  Loan_ID = test$loan_id,
  Loan_Status = xg_preds
) 
xg_pred
write_csv(xg_pred,"boost2.csv")
