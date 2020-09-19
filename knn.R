lg_mod <- logistic_reg(penalty = tune(),
                       mixture = tune()) %>% 
  set_engine("glm")

lg_wkflw <- workflow() %>% 
  add_recipe(rec_prep) %>% 
  add_model(lg_mod)
lg_grid <- grid_random(penalty(),
                       mixture(),
                       size = 20)

lg_tune <- tune_grid(lg_wkflw,
                     resamples = folds,
                     grid = lg_grid,
                     metrics = metric_set(accuracy),
                     control = control_grid(save_pred = TRUE))
lg_tune %>% collect_metrics()
lg_tune %>% show_best()

lg_params <- lg_tune %>% select_best()
lg_params
lg_workflow <- lg_wkflw %>% finalize_workflow(lg_params)
lg_fit <- lg_workflow %>% last_fit(train_split)
lg_fit %>% collect_metrics()
lg_final <- lg_workflow %>% fit(train)
lg_final
lg_preds <- lg_final %>% predict(test)
lg_preds
