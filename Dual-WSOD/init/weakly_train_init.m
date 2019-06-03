function save_model_path = weakly_train_init(image_roidb_train, model, cache_name, box_param, rng_seed)
% --------------------------------------------------------
% Dual-Network implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs

    %ip.parse(imdb_train, roidb_train, varargin{:});
    assert(isfield(box_param, 'bbox_means'));
    assert(isfield(box_param, 'bbox_stds'));
    assert(isfield(model, 'solver_def_file'));
    assert(isfield(model, 'test_net_def_file'));
    assert(isfield(model, 'net_file'));
    assert(isfield(model, 'name'));
    assert(isfield(model, 'conf'));
    assert(isfield(model.conf, 'classes'));
    assert(isfield(model.conf, 'max_epoch'));
    assert(isfield(model.conf, 'step_epoch'));
    assert(isfield(model.conf, 'regression'));

%% try to find trained model
    cache_dir       = fullfile(pwd, 'output', 'weakly_cachedir', cache_name, 'initial');
    debug_cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', cache_name, 'debug');
    classes = model.conf.classes;
%% init
    % set random seed
    prev_rng = seed_rand(rng_seed);
    caffe.set_random_seed(rng_seed);
    
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
    
    disp('conf:');
    disp(model.conf);

    %train_modes = cell(numel(models), 1);
    %for idx = 1:numel(models)
    %  train_modes{idx} = weakly_train_mode(models{idx}.conf);
    %  fprintf('conf: %2d : %s :: train mode : %d\n', idx, models{idx}.name, train_modes{idx});
    %  disp(models{idx}.conf);
    %end
    train_mode = weakly_train_mode(model.conf);

%% training
    model_suffix   = '.caffemodel';
    model.cur_net_file = weakly_supervised(train_mode, image_roidb_train, model.solver_def_file, model.net_file, 1000, ...
                                               box_param, model.conf, cache_dir, [model.name, '_init'], model_suffix, 'final');

    fprintf('Final [%s] Model Path : %s\n', model.name, model.cur_net_file);
    save_model_path = model.cur_net_file;

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
