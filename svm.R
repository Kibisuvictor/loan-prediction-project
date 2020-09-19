?svm_poly
sv_mod <- svm_poly(mode = "classification",
                   cost = tune(),
                   degree = tune(),
                   scale_factor = tune()) %>% 
  set_engine("kernlab")

sv_wkflw <- workflow() %>% 
  add_recipe(rec_prep) %>% 
  add_model(sv_mod)

sv_grid <- grid_random(cost(),
                       degree(),
                       scale_factor(),
                       size = 20)

sv_tune <- tune_grid(sv_wkflw,
                     resamples = folds,
                     grid = sv_grid,
                     metrics = metric_set(accuracy),
                     control = control_grid(save_pred = TRUE))
sv_tune %>% collect_metrics()
sv_tune %>% show_best()
sv_params <- sv_tune %>% select_best()
sv_workflow <- sv_wkflw %>% finalize_workflow(sv_params)
sv_fit <- sv_workflow %>% last_fit(train_split)
sv_fit %>% collect_metrics()
sv_final <- 