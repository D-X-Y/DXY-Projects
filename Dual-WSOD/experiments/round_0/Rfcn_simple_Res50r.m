figure('visible', 'off');
clear; close all; clc;
clear mex;
clear is_valid_handle; % to clear init_key
%%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
%opts.gpu_id                 = 3; %% GPU-ID 1
%opts.gpu_id                 = 4; %% GPU-ID 0
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
models{1}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
models{1}.cur_net_file      = 'unset';
models{1}.name              = 'Res50r-OHEM-Rfcn';
models{1}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image.mat');
models{1}.conf              = rfcn_config_ohem('image_means', models{1}.mean_image, ... 
                                               'classes', classes, ... 
                                               'max_epoch', 4, 'step_epoch', 2, ... 
                                               'regression', true);
box_param{1}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat'));

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
opts.cache_name             = [opts.cache_name, '-nms_', num2str(nms_thresh), '_', num2str(cof_thresh), '-seed_', num2str(rng_seed), ...
                                    '-l_', num2str(models{1}.conf.step_epoch), '_', num2str(models{1}.conf.max_epoch), '-', difficult_string];

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

b_image_roidb_train     = get_init_filter(models{1}.conf, a_image_roidb_train, box_param{1});

opts.rfcn_model         = weakly_train_init(b_image_roidb_train, models{1}, opts.cache_name, box_param{1}, rng_seed);


fprintf('-------------------- TESTING --------------------\n');
test_time               = tic;
[mAP, meanloc]          = weakly_test_all({models{1}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{1}.test_net_def_file}, ...
                                'net_models',       {opts.rfcn_model}, ...
                                'log_prefix',       'final', ...
                                'cache_name',       opts.cache_name,...
                                'ignore_cache',     true);

mAPs       = zeros(models{1}.conf.max_epoch, 1);
CorLocs    = zeros(models{1}.conf.max_epoch, 1);
rfcn_model = cell (models{1}.conf.max_epoch, 1);
for epoch = 1:models{1}.conf.max_epoch
    rfcn_model{epoch} = fullfile(pwd, 'output', 'weakly_cachedir' , opts.cache_name, 'initial', [models{1}.name, '_init_epoch_', num2str(epoch), '.caffemodel']);
    assert(exist(rfcn_model{epoch}, 'file') ~= 0, 'not found trained model');
end
for epoch = 1:models{1}.conf.max_epoch
[mAPs(epoch), CorLocs(epoch)] = weakly_test_all({models{1}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                  'net_defs',         {models{1}.test_net_def_file}, ...
                                  'net_models',       rfcn_model(epoch), ...
                                  'cache_name',       opts.cache_name,...
                                  'log_prefix',       ['epoch_', num2str(epoch)], ...
                                  'ignore_cache',     true);
end
