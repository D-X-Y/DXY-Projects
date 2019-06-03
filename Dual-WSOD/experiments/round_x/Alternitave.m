figure('visible', 'off');
clear; close all; clc;
clear mex;
clear is_valid_handle; % to clear init_key
%%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu();
active_caffe_mex(opts.gpu_id, opts.caffe_version);
fprintf('Gpu config done : %d\n', opts.gpu_id);

% global parameters
extra_para                  = load(fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat'));
classes                     = extra_para.VOCopts.classes;
print_result                = true;
use_flipped                 = true;
exclude_difficult_samples   = false;
rng_seed                    = 5;
% model
models                      = cell(1,1);
box_param                   = cell(1,1);
models{1}.solver_def_file   = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'resnet50_res5', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'resnet50_res5', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
models{1}.cur_net_file      = 'unset';
models{1}.name              = 'Res50r-SIMPLE-Fast';
models{1}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image.mat');
models{1}.conf              = fast_rcnn_config('image_means', models{1}.mean_image, ...
                                               'classes', classes, ...
                                               'max_epoch', 4, 'step_epoch', 2, ...
                                               'regression', true, ...
                                               'max_rois_num_in_gpu',  1000);
box_param{1}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'fast_box_param.mat'));

models{2}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'solver_lr1_3.prototxt');
models{2}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'test.prototxt');
models{2}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
models{2}.cur_net_file      = 'unset';
models{2}.name              = 'Res50r-OHEM-Rfcn';
models{2}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image.mat');
models{2}.conf              = rfcn_config_ohem('image_means', models{2}.mean_image, ... 
                                               'classes', classes, ... 
                                               'max_epoch', 4, 'step_epoch', 2, ... 
                                               'regression', true);
box_param{2}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat'));

% cache name
path_col                    = '/home/xuanyi/data/VOC2007.h5';
opts.cache_name             = ['INIT-', models{1}.name]
%[1 : 29.5] [2 : 33.7] [3 : 35.3]
nms_thresh                  = 0.3;
cof_thresh                  = 0.1;
if (exclude_difficult_samples == false)
  difficult_string            = 'difficult';
else
  difficult_string            = 'easy';
end
opts.cache_name             = [opts.cache_name, '-nms_', num2str(nms_thresh), '-', num2str(cof_thresh), '-seed_', num2str(rng_seed), ...
                                          '-l_', num2str(models{1}.conf.step_epoch), '_', num2str(models{1}.conf.max_epoch),  '-', difficult_string];

% dataset
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', use_flipped, exclude_difficult_samples);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false, exclude_difficult_samples);
imdbs_name                  = cell2mat(cellfun(@(x) x.name, dataset.imdb_train,'UniformOutput', false));


mkdir_if_missing(['output/', 'weakly_cachedir/', opts.cache_name]);
save_file = fullfile('output', 'weakly_cachedir', opts.cache_name, ['voc2007_corloc_nms', '-', num2str(nms_thresh), '-', num2str(cof_thresh), '.mat']);
try
  load(save_file);
  fprintf('Load pre saved data from %s\n', save_file);
catch
% train/test data
  tic;
  fprintf('--- Fail load %s Loading dataset ... \n', save_file);
  fprintf('-------------------- INITIALIZATION --------------------\n');
  image_roidb_train         = generate_pure(models{1}.conf, dataset.imdb_train, dataset.roidb_train, box_param{1}, use_flipped, path_col);
  fprintf('--- Generate image_roidb_train done in %.1fs\n', toc);
  image_roidb_test          = generate_pure(models{1}.conf, dataset.imdb_test,  dataset.roidb_test,  box_param{1},       false, path_col);
  fprintf('--- Generate image_roidb_test  done in %.1f s\n', toc);
  save(save_file, 'image_roidb_train', 'image_roidb_test', 'imdbs_name', '-v7.3');
  fprintf('--- Save data into %s, cost %.1f\n', save_file, toc);
end
save_file = fullfile('output', 'weakly_cachedir', opts.cache_name, ['voc2007_corloc_nms', '-', num2str(nms_thresh), '-', num2str(cof_thresh), '-aaa.mat']);
try
  load(save_file);
  fprintf('Load pre saved a_image_roidb_train from %s\n', save_file);
catch
  tic;
  a_image_roidb_train     = get_init_pesudo(models{1}.conf, image_roidb_train, nms_thresh, cof_thresh);
  save(save_file, 'a_image_roidb_train', '-v7.3');
  fprintf('--- Save a_image_roidb_train into %s, cost %.1f s\n', save_file, toc);
end

O_image_roidb_train     = get_init_filter(models{1}.conf, a_image_roidb_train, box_param{1});
T_image_roidb_train     = get_init_filter(models{2}.conf, a_image_roidb_train, box_param{2});

models{1}.name            = 'Res50r-SIMPLE-Fast-Stage1';
stage1_fast               = weakly_train_init(O_image_roidb_train, models{1}, opts.cache_name, box_param{1}, rng_seed);

models{2}.name            = 'Res50r-OHEM-Rfcn-Stage1';
models{2}.net_file        = stage1_fast;
models{2}.conf.max_epoch  = 4; 
models{2}.conf.step_epoch = 2;
stage1_rfcn               = weakly_train_init(T_image_roidb_train, models{2}, opts.cache_name, box_param{2}, rng_seed);


models{1}.name            = 'Res50r-SIMPLE-Fast-Stage2';
models{1}.net_file        = stage1_rfcn;
models{1}.conf.max_epoch  = 4; 
models{1}.conf.step_epoch = 2;
stage2_fast               = weakly_train_init(O_image_roidb_train, models{1}, opts.cache_name, box_param{1}, rng_seed);


models{2}.name            = 'Res50r-OHEM-Rfcn-Stage2';
models{2}.net_file        = stage2_fast;
models{2}.conf.max_epoch  = 2; 
models{2}.conf.step_epoch = 1;
models{2}.solver_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'solver_lr1_3.fix');
stage2_rfcn               = weakly_train_init(T_image_roidb_train, models{2}, opts.cache_name, box_param{2}, rng_seed);


fprintf('-------------------- TESTING --------------------\n');
test_time               = tic;
mAP                     = zeros(4,1);
meanloc                 = zeros(4,1);
[mAP(1), meanloc(1)]    = weakly_test_all({models{1}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{1}.test_net_def_file}, ...
                                'net_models',       {stage1_fast}, ...
                                'log_prefix',       'fast_stage_1', ...
                                'cache_name',       opts.cache_name,...
                                'dis_itertion',     500, ...
                                'ignore_cache',     true);

[mAP(2), meanloc(2)]    = weakly_test_all({models{2}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{2}.test_net_def_file}, ...
                                'net_models',       {stage1_rfcn}, ...
                                'log_prefix',       'rfcn_stage_1', ...
                                'cache_name',       opts.cache_name,...
                                'dis_itertion',     500, ...
                                'ignore_cache',     true);

[mAP(3), meanloc(3)]    = weakly_test_all({models{1}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{1}.test_net_def_file}, ...
                                'net_models',       {stage2_fast}, ...
                                'log_prefix',       'fast_stage_2', ...
                                'cache_name',       opts.cache_name,...
                                'dis_itertion',     500, ...
                                'ignore_cache',     true);

[mAP(4), meanloc(4)]    = weakly_test_all({models{2}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{2}.test_net_def_file}, ...
                                'net_models',       {stage2_rfcn}, ...
                                'log_prefix',       'rfcn_stage_2', ...
                                'cache_name',       opts.cache_name,...
                                'dis_itertion',     500, ...
                                'ignore_cache',     true);
