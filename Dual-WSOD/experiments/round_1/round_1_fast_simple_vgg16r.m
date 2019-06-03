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
models{1}.solver_def_file   = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'VGG16', 'vgg16.caffemodel');
models{1}.cur_net_file      = fullfile(pwd, 'models', 'trained', 'VGG16r-SIMPLE-Fast_init_epoch_3.caffemodel');
models{1}.name              = 'VGG16r-SIMPLE-Fast';
models{1}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'VGG16', 'mean_image.mat');
models{1}.conf              = fast_rcnn_config('image_means', models{1}.mean_image, ... 
                                               'classes', classes, ... 
                                               'max_epoch', 4, 'step_epoch', 2, ... 
                                               'regression', true);
box_param{1}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'fast_box_param.mat'));

if (exclude_difficult_samples == false)
  difficult_string            = 'difficult';
else
  difficult_string            = 'easy';
end
opts.cache_name             = ['R1-seed_', num2str(rng_seed), '-', difficult_string];

% dataset
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', use_flipped, exclude_difficult_samples);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false, exclude_difficult_samples);
imdbs_name                  = cell2mat(cellfun(@(x) x.name, dataset.imdb_train,'UniformOutput', false));


mkdir_if_missing(['output/', 'weakly_cachedir/', opts.cache_name]);
cache_dir       = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, imdbs_name);
debug_cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug');

temp = load (fullfile(pwd, 'output', 'weakly_cachedir', 'voc2007_corloc_nms-0.3-0.1-trainval.mat'));
base_select = 3;
image_roidb_train = temp.image_roidb_train;
pre_keeps   = zeros(numel(image_roidb_train), numel(models));
gamma       = 0.3;

[previous_model, pre_keeps] = weakly_dual_train_step(image_roidb_train, models, box_param, base_select, pre_keeps, gamma, rng_seed, cache_dir, debug_cache_dir);

opts.rfcn_model = previous_model;


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
    rfcn_model{epoch} = fullfile(pwd, 'output', 'weakly_cachedir' , opts.cache_name, 'initial', [models{1}.name, '_epoch_', num2str(epoch), '.caffemodel']);
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
