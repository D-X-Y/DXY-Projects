function save_model_path = weakly_co_train_v3(imdb_train, roidb_train, models, varargin)
% --------------------------------------------------------
% MSPLD implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    %ip.addRequired('conf',                              @isstruct);
    assert (iscell(imdb_train));
    assert (iscell(roidb_train));
    %ip.addRequired('imdb_train',                        @iscell);
    %ip.addRequired('roidb_train',                       @iscell);
    ip.addParamValue('per_class_sample',    3,          @isscalar);
    ip.addParamValue('val_interval',      500,          @isscalar); 
    ip.addParamValue('base_select',       [1],          @isvector); 
    ip.addParamValue('rng_seed',            5,          @isscalar); 
    ip.addParamValue('gamma',             0.3,          @isscalar); 
    ip.addParamValue('debug',           false,          @islogical); 
    ip.addParamValue('use_flipped',      true,          @islogical); 
    ip.addParamValue('boost',           false,          @islogical); 
    ip.addParamValue('cache_name',        'un-define',  @isstr);
    ip.addParamValue('box_param',         cell(0,0),    @iscell);

    %ip.parse(imdb_train, roidb_train, varargin{:});
    ip.parse(varargin{:});
    opts = ip.Results;
    assert(iscell(models));
    assert(isfield(opts, 'box_param'));
    for i = 1:numel(opts.box_param)
        assert(isfield(opts.box_param{i}, 'bbox_means'));
        assert(isfield(opts.box_param{i}, 'bbox_stds'));
    end
    assert(numel(models) == numel(opts.box_param));
    for idx = 1:numel(models)
        assert(isfield(models{idx}, 'solver_def_file'));
        assert(isfield(models{idx}, 'test_net_def_file'));
        assert(isfield(models{idx}, 'net_file'));
        assert(isfield(models{idx}, 'name'));
        assert(isfield(models{idx}, 'conf'));
        assert(isfield(models{idx}.conf, 'classes'));
        assert(isfield(models{idx}.conf, 'max_epoch'));
        assert(isfield(models{idx}.conf, 'step_epoch'));
        assert(isfield(models{idx}.conf, 'regression'));
    end
    
%% try to find trained model
    imdbs_name      = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir       = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, imdbs_name);
    debug_cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug');
    classes = models{1}.conf.classes;
    if (numel(opts.per_class_sample) == 1)
        opts.per_class_sample = opts.per_class_sample * ones(numel(classes), 1);
    end
%% init
    % set random seed
    prev_rng = seed_rand(opts.rng_seed);
    caffe.set_random_seed(opts.rng_seed);
    
    % init caffe solver
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);

    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['co_train_', timestamp, '.txt']);
    diary(log_file);

    % set gpu mode, mush run on gpu
    caffe.reset_all();
    caffe.set_mode_gpu(); 
    
    disp('opts:');
    disp(opts);

    train_modes = cell(numel(models), 1);
    for idx = 1:numel(models)
      train_modes{idx} = weakly_train_mode(models{idx}.conf);
      fprintf('conf: %2d : %s :: train mode : %d\n', idx, models{idx}.name, train_modes{idx});
      disp(models{idx}.conf);
    end
    

%% making tran/val data
    fprintf('Preparing training data...');
    image_roidb_train = cellfun(@(x, y) ... // @(imdbs, roidbs)
                            arrayfun(@(z) ... //@([1:length(x.image_ids)])
                                struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                                'overlap', y.rois(z).overlap, 'boxes', y.rois(z).boxes, 'class', y.rois(z).class, 'image', [], 'bbox_targets', [], ...
                                'GT_Index', y.rois(z).gt, 'image_label', setdiff(unique(y.rois(z).class(y.rois(z).gt)), [0]), 'Debug_GT_Cls', [], 'Debug_GT_Box', []), ...
                                [1:length(x.image_ids)]', 'UniformOutput', true), imdb_train(:), roidb_train(:), 'UniformOutput', false);
    image_roidb_train = cat(1, image_roidb_train{:});

    [warmup_ids, unsupervise_ids] = weakly_sample_train_v2(image_roidb_train, opts.per_class_sample, imdb_train{1}.flip);

    warmup_roidb_train = cell(numel(models), 1);
    image_roidb_train  = cell(numel(models), 1);
    for i = 1:numel(models)
        cur_image_roidb_train = weakly_prepare_image_roidb(models{i}.conf, imdb_train, roidb_train, opts.box_param{i}.bbox_means, opts.box_param{i}.bbox_stds);
        warmup_roidb_train{i} = clean_data(cur_image_roidb_train(warmup_ids), true);
        image_roidb_train{i}  = clean_data(cur_image_roidb_train(unsupervise_ids), false);
    end
    fprintf('Done.\n');
%% assert conf flip attr
    for idx = 1:numel(imdb_train)
        assert ( imdb_train{idx}.flip == opts.use_flipped);
    end

%% training
    model_suffix   = '.caffemodel';
    previous_model = cell(numel(models), 1);
    for idx = 1:numel(models)
        previous_model{idx} = weakly_supervised(train_modes{idx}, warmup_roidb_train{idx}, models{idx}.solver_def_file, models{idx}.net_file, opts.val_interval, ...
                                                opts.box_param{idx}, models{idx}.conf, cache_dir, [models{idx}.name, '_Loop_0'], model_suffix, 'final');
        models{idx}.cur_net_file = previous_model{idx};
    end

    %pre_keep = false(numel(unsupervise_ids), 1);
    pre_keeps = zeros(numel(unsupervise_ids), numel(models));

    % rng_seed 5; per_sample 3;
    Init_Per_Select = [40, 12, 10, 20, 20, 10, 50, 25, 15, 10,...
                       20, 15, 15, 30, 15, 15, 15, 20, 40, 35];
%% Start Training
    total_loop = numel(opts.base_select);
    total_model = numel(models);
    for index = 1:numel(opts.base_select)

        base_select = opts.base_select(index);
        fprintf('\n-------Start Loop [%2d]/[%2d] <---> with base_select : %4.2f-------\n', index, total_loop, base_select);
        [A_image_roidb_train] = weakly_generate_pseudo(models, image_roidb_train{1}, opts.boost);

        for idx = 1:numel(models)
            fprintf('>>>>>>>>For [%2d]/[%2d]-th model: %s\n', idx, total_model, models{idx}.name);
            PER_Select = ceil(Init_Per_Select * base_select);
            %% Filter Unreliable Image with pseudo-boxes
            [B_image_roidb_train] = weakly_clean_data(classes, A_image_roidb_train, 15);
            [B_image_roidb_train] = weakly_full_targets(models{idx}.conf, B_image_roidb_train, opts.box_param{idx}.bbox_means, opts.box_param{idx}.bbox_stds);

            diff_set = setdiff((1:numel(models)), idx);
            pre_keep = sum(pre_keeps(:,diff_set), 2);
            [C_image_roidb_train] = weakly_filter_loss(models{idx}, B_image_roidb_train, pre_keep, 0.6, opts.gamma);

            [D_image_roidb_train] = weakly_filter_score(models, C_image_roidb_train, PER_Select);
            [D_image_roidb_train] = weakly_full_targets(models{idx}.conf, D_image_roidb_train, opts.box_param{idx}.bbox_means, opts.box_param{idx}.bbox_stds);

            %pre_keep = false(numel(unsupervise_ids), 1);
            for j = 1:numel(D_image_roidb_train) , pre_keeps(D_image_roidb_train(j).index, idx) = true; end
            if (opts.debug), inloop_debug(D_image_roidb_train, classes, debug_cache_dir, ['L_', num2str(index), '_', models{idx}.name, '_D']); end

            new_image_roidb_train = [warmup_roidb_train{idx}; D_image_roidb_train];
            
            train_mode = weakly_train_mode (models{idx}.conf);
            previous_model{idx}   = weakly_supervised(train_mode, new_image_roidb_train, models{idx}.solver_def_file, models{idx}.net_file, opts.val_interval, ...
                                                      opts.box_param{idx}, models{idx}.conf, cache_dir, [models{idx}.name, '_Loop_', num2str(index)], model_suffix, 'final');

        end

        for idx = 1:numel(models)
            models{idx}.cur_net_file = previous_model{idx};
        end
    end

    save_model_path    = cell(numel(models), 1);
    for idx = 1:numel(models)
        weakly_final_model   = sprintf('%s_final%s', models{idx}.name, model_suffix);
        weakly_final_model   = fullfile(cache_dir, weakly_final_model);
        fprintf('Final [%s] Model Path : %s\n', models{idx}.name, weakly_final_model);
        save_model_path{idx} = weakly_final_model;
        copyfile(previous_model{idx}, save_model_path{idx});
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

function select = inloop_count(conf, image_roidb_train)
  numcls = numel(conf.classes);
  select = zeros(numcls, 1);
  for index = 1:numel(image_roidb_train)
    class = image_roidb_train(index).image_label;
    for j = 1:numel(class)
      select(class(j)) = select(class(j)) + 1;
    end
  end
end

function x = clean_data(image_roidb_train, save_overlap)
  x = [];
  for index = 1:numel(image_roidb_train)
    gt = image_roidb_train(index).GT_Index;
    if (save_overlap == false)
        Struct = struct('image_path',  image_roidb_train(index).image_path, ...
                        'image_id',    image_roidb_train(index).image_id, ...
                        'imdb_name',   image_roidb_train(index).imdb_name, ...
                        'im_size',     image_roidb_train(index).im_size, ...
                        'overlap',     [], ...
                        'boxes',       image_roidb_train(index).boxes(~gt, :), ...
                        'bbox_targets', [], ...
                        'pseudo_boxes', [], ...
                        'Debug_GT_Cls', image_roidb_train(index).class(gt, :), ...
                        'Debug_GT_Box', image_roidb_train(index).boxes(gt, :), ...
                        'image_label', image_roidb_train(index).image_label, ...
                        'index', index);
    else
        Struct = struct('image_path',  image_roidb_train(index).image_path, ...
                        'image_id',    image_roidb_train(index).image_id, ...
                        'imdb_name',   image_roidb_train(index).imdb_name, ...
                        'im_size',     image_roidb_train(index).im_size, ...
                        'overlap',     image_roidb_train(index).overlap, ...
                        'boxes',       image_roidb_train(index).boxes, ...
                        'bbox_targets', image_roidb_train(index).bbox_targets, ...
                        'pseudo_boxes', [], ...
                        'Debug_GT_Cls', image_roidb_train(index).class(gt, :), ...
                        'Debug_GT_Box', image_roidb_train(index).boxes(gt, :), ...
                        'image_label',  image_roidb_train(index).image_label, ...
                        'index', index);
    end
    x{end+1} = Struct;
  end
  if numel(x) ~= 0
    x = cat(1, x{:});
  end
end
