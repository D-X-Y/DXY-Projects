function [previous_model, next_keeps] = weakly_dual_train_step(image_roidb_train, models, box_params, base_select, pre_keeps, gamma, class_limit, rng_seed, cache_dir, debug_cache_dir)
% --------------------------------------------------------
% Dual-Network implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    assert (numel(models) <= 2);
    assert (numel(models) == numel(box_params));
    for idx = 1:numel(box_params)
        assert(isfield(box_params{idx}, 'bbox_means'));
        assert(isfield(box_params{idx}, 'bbox_stds'));
        assert(isfield(models{idx}, 'conf'));
        assert(isfield(models{idx}, 'solver_def_file'));
        assert(isfield(models{idx}, 'test_net_def_file'));
        assert(isfield(models{idx}, 'net_file'));
        assert(isfield(models{idx}, 'cur_net_file'));
        assert(isfield(models{idx}, 'name'));
        assert(isfield(models{idx}, 'conf'));
        assert(isfield(models{idx}.conf, 'classes'));
        assert(isfield(models{idx}.conf, 'max_epoch'));
        assert(isfield(models{idx}.conf, 'step_epoch'));
        assert(isfield(models{idx}.conf, 'regression'));
        assert(isfield(models{idx}.conf, 'fast_rcnn'));
    end
%% initialize some varibles
    classes = models{1}.conf.classes;

    prev_rng = seed_rand(rng_seed);
    caffe.set_random_seed(rng_seed);
    
    mkdir_if_missing(cache_dir);
    mkdir_if_missing(debug_cache_dir);

    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);


    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['dual_train_one_step_', timestamp, '.txt']);
    diary(log_file);

    caffe.reset_all();
    caffe.set_mode_gpu(); 
    
    train_modes = cell(numel(models), 1);
    fprintf('cache_dir       : %s\n', cache_dir);
    fprintf('debug_cache_dir : %s\n', debug_cache_dir);
    fprintf('gamma : %.3f  ,  rng_seed : %.3f\n', gamma, rng_seed);
    for idx = 1:numel(models)
      train_modes{idx} = weakly_train_mode(models{idx}.conf);
      fprintf('conf: %2d : %s :: train mode : %d\n', idx, models{idx}.name, train_modes{idx});
      disp(models{idx}.conf);
    end
    

%% making tran/val data
    %fprintf('Preparing training data...');
    %image_roidb_train  = clean_data(ori_image_roidb_train);
    %fprintf('Done.\n');

%% training
    model_suffix   = '.caffemodel';
    previous_model = cell(numel(models), 1);

%% Start Training
    P_image_roidb_train = [];
    for i = 1:numel(image_roidb_train)
      image_label = image_roidb_train(i).image_label;
      image_label = unique(image_label);
      if (numel(image_label) <= class_limit)
        P_image_roidb_train{end+1} = image_roidb_train(i);
      end
    end
    P_image_roidb_train = cat(1, P_image_roidb_train{:});

    fprintf('\n-------Start Loop with total base_select : %4d----- %4d -> %5d\n', sum(base_select), numel(image_roidb_train), numel(P_image_roidb_train));
    [A_image_roidb_train] = weakly_generate_pseudo(models, P_image_roidb_train);

    loss_save_ratio = 0.95;
    base_select = base_select ./ loss_save_ratio;
    next_keeps = false(size(pre_keeps));
    for idx = 1:numel(models)
        fprintf('>>>>>>>>For %3d : %s\n', idx, models{idx}.name);

        %% Design select how much
        pre_base_select = inloop_cal_num(A_image_roidb_train, classes);
        if (sum(pre_base_select) >= sum(base_select))
           pre_base_select = ceil( pre_base_select * sum(base_select) / sum(pre_base_select) ) + 1;
        end 
        pre_base_select = max([base_select;pre_base_select]);

        [S_image_roidb_train] = weakly_filter_score(models, A_image_roidb_train, pre_base_select);
        [S_image_roidb_train] = weakly_full_targets(models{idx}.conf, S_image_roidb_train, box_params{idx}.bbox_means, box_params{idx}.bbox_stds);

        inloop_debug(S_image_roidb_train, classes, debug_cache_dir, [models{idx}.name, '_S']);

        if (numel(models) == 2),     pre_keep = pre_keeps(:,3-idx);
        elseif (numel(models) == 1), pre_keep = pre_keeps;
        else,                        assert(false); end
        [L_image_roidb_train] = weakly_filter_loss (models{idx}, S_image_roidb_train, pre_keep, loss_save_ratio, gamma);

        %% Filter Unreliable Image with pseudo-boxes
        %[B_image_roidb_train] = weakly_clean_data(classes, A_image_roidb_train, 15);
        %[D_image_roidb_train] = weakly_filter_score(models, C_image_roidb_train, pre_base_select);
        %[D_image_roidb_train] = weakly_full_targets(models{idx}.conf, D_image_roidb_train, box_params{idx}.bbox_means, box_params{idx}.bbox_stds);

        %pre_keep = false(numel(unsupervise_ids), 1);
        for j = 1:numel(L_image_roidb_train), next_keeps(L_image_roidb_train(j).index, idx) = true; end
        %if (opts.debug), inloop_debug(D_image_roidb_train, classes, debug_cache_dir, ['L_', num2str(index), '_', models{idx}.name, '_D']); end

        new_image_roidb_train = L_image_roidb_train;
            
        previous_model{idx}   = weakly_supervised(train_modes{idx}, new_image_roidb_train, models{idx}.solver_def_file, models{idx}.cur_net_file, 1000, ...
                                                      box_params{idx}, models{idx}.conf, cache_dir, models{idx}.name, model_suffix, 'final');
    end

    for idx = 1:numel(models)
        fprintf('Final [%s] Model Path : %s\n', models{idx}.name, previous_model{idx});
    end

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end

function inloop_debug(image_roidb_train, classes, debug_cache_dir, dir_name)
  debug_cache_dir = fullfile(debug_cache_dir, dir_name);
  for iii = 1:numel(image_roidb_train)
    if (strcmp('flip', image_roidb_train(iii).image_id(end-3:end)) == 1), continue; end
    weakly_debug_final(classes, debug_cache_dir, image_roidb_train(iii));
  end
end

function temp_num = inloop_cal_num(T_image_roidb_train, classes)
  temp_num = zeros(numel(classes), 1);
  for i = 1:numel(T_image_roidb_train)
    image_label = T_image_roidb_train(i).image_label;
    for j = 1:numel(image_label)
      temp_num(image_label(j)) = temp_num(image_label(j)) + 1;
    end
  end
  temp_num = temp_num';
end
