figure('visible', 'off');
clear; close all; clc;
clear mex;
clear is_valid_handle; % to clear init_key
%%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = 3;%auto_select_gpu();
active_caffe_mex(opts.gpu_id, opts.caffe_version);
fprintf('Gpu config done : %d\n', opts.gpu_id);

% global parameters
extra_para                  = load(fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat'));
classes                     = extra_para.VOCopts.classes;
print_result                = true;
use_flipped                 = true;
exclude_difficult_samples   = false;
class_limit                 = 1;
ctime                       = datestr(now, 30); %取系统时间
tseed                       = str2num(ctime((end - 5):end)); %将时间字符转换为数字
rand('seed', tseed);
rng_seed                    = randi(10000);
% model
models                      = cell(2,1);
box_param                   = cell(2,1);

models{2}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'solver_lr1_3.prototxt');
models{2}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'test.prototxt');
models{2}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
models{2}.cur_net_file      = fullfile(pwd, 'models', 'trained', 'Res50r-OHEM-Rfcn_init_final.caffemodel');
models{2}.name              = 'Res50r-OHEM-Rfcn';
models{2}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image.mat');
models{2}.conf              = rfcn_config_ohem('image_means', models{2}.mean_image, ... 
                                               'classes', classes, ... 
                                               'max_epoch', 4, 'step_epoch', 2, ... 
                                               'regression', true);
box_param{2}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat'));

models{1}.solver_def_file   = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'resnet50_res5', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'resnet50_res5', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
models{1}.cur_net_file      = fullfile(pwd, 'models', 'trained', 'Res50r-SIMPLE-Fast-Stage1_init_final.caffemodel');
models{1}.name              = 'Res50r-SIMPLE-Fast';
models{1}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image.mat');
models{1}.conf              = fast_rcnn_config('image_means', models{1}.mean_image, ...
                                               'classes', classes, ...
                                               'max_epoch', 4, 'step_epoch', 2, ...
                                               'regression', true, ...
                                               'max_rois_num_in_gpu', 1200);
box_param{1}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'fast_box_param.mat'));

% cache name
if (exclude_difficult_samples == false)
  difficult_string          = 'difficult';
else
  difficult_string          = 'easy';
end
opts.cache_name             = ['T07-seed_', num2str(rng_seed), '-', models{1}.name, '-', models{2}.name, '-L', num2str(class_limit), '-Rng', num2str(rng_seed), '-', difficult_string];

% dataset
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', false, false);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false, false);
imdbs_name                  = cell2mat(cellfun(@(x) x.name, dataset.imdb_train,'UniformOutput', false));

mkdir_if_missing(['output/', 'weakly_cachedir/', opts.cache_name]);
cache_dir       = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, imdbs_name);
debug_cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug');

test_time                   = tic;
fprintf('-------------------- TESTING STEP 1 --------------------\n');
step1_models                = cell(2,1);
step1_models{1}             = models{1}.cur_net_file;
step1_models{2}             = models{2}.cur_net_file;

corloc                      = weakly_test_Cor_v2({models{1}.conf, models{2}.conf}, dataset.imdb_train{1}, dataset.roidb_train{1}, ... 
                                'net_defs',        {models{1}.test_net_def_file, models{2}.test_net_def_file}, ... 
                                'net_models',      step1_models, ... 
                                'cache_name',      opts.cache_name, ...
                                'log_prefix',      'co_cor_final', ...
                                'ignore_cache',    true);
