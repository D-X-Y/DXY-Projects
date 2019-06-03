function model_path = weakly_supervised(train_mode, roidb_train, solver_file, model_file, val_interval, box_param, conf, cache_dir, prefix, suffix, final_name)
    if    (train_mode == 0)
        model_path =  weakly_supervised_rfcn(roidb_train, solver_file, model_file, val_interval, box_param, conf, cache_dir, prefix, suffix, final_name);
    elseif(train_mode == 1)
        model_path = weakly_supervised_score(roidb_train, solver_file, model_file, val_interval,            conf, cache_dir, prefix, suffix, final_name);
    elseif(train_mode == 2)
        model_path =  weakly_supervised_fast(roidb_train, solver_file, model_file, val_interval, box_param, conf, cache_dir, prefix, suffix, final_name);
    end
end
