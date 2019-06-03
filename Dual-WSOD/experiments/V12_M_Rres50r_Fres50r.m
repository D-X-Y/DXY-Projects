figure('visible', 'off');
clear; close all; clc;
clear mex;
clear is_valid_handle; % to clear init_key
%%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = 1;%auto_select_gpu();
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
                                               'max_rois_num_in_gpu', 1400);
box_param{1}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'fast_box_param.mat'));

% cache name
if (exclude_difficult_samples == false)
  difficult_string          = 'difficult';
else
  difficult_string          = 'easy';
end
opts.cache_name             = ['V12-seed_', num2str(rng_seed), '-', models{1}.name, '-', models{2}.name, '-L', num2str(class_limit), '-Rng', num2str(rng_seed), '-', difficult_string];

% dataset
dataset                     = [];
dataset                     = Dataset.voc0712_trainval_ss(dataset, 'train', use_flipped, false);
dataset                     = Dataset.voc2012_test2online_ss(dataset, 'test', false);
imdbs_name                  = cell2mat(cellfun(@(x) x.name, dataset.imdb_train,'UniformOutput', false));

[image_roidb_train]         = weakly_prepare_image_roidb(models{1}.conf, dataset.imdb_train, dataset.roidb_train, box_param{1}.bbox_means, box_param{1}.bbox_stds);
filtered_image_roidb_train = [];
for index = 1:numel(image_roidb_train)
   gt = image_roidb_train(index).GT_Index;
   is_flip = false;
   if (use_flipped == false)
        flip = -1;
        image_id = image_roidb_train(index).image_id;
   elseif (strcmp(image_roidb_train(index).image_id(end-3:end), 'flip') == 1)
        flip = index-1;
        is_flip = true;
        assert (strcmp([image_roidb_train(flip).image_id, '_flip'], image_roidb_train(index).image_id) == 1);
        image_id = image_roidb_train(index).image_id(1:end-5);
   else
        flip = index+1;
        assert (strcmp([image_roidb_train(index).image_id, '_flip'], image_roidb_train(flip).image_id) == 1);
        image_id = image_roidb_train(index).image_id;
   end

   Struct = struct('image_path', image_roidb_train(index).image_path, ...
                   'image_id',   image_roidb_train(index).image_id, ...
                   'imdb_name',  image_roidb_train(index).imdb_name, ...
                   'im_size',    image_roidb_train(index).im_size, ...
                   'overlap',    [], ...
                   'boxes',      image_roidb_train(index).boxes(~gt, :), ...
                   'bbox_targets', [], ...
                   'pseudo_boxes', [], ...
                   'Debug_GT_Cls', image_roidb_train(index).class(gt, :), ...
                   'Debug_GT_Box', image_roidb_train(index).boxes(gt, :), ...
                   'image_label',  image_roidb_train(index).image_label, ...
                   'flipid',       flip, ...
                   'is_flip',      is_flip, ...
                   'index',        index);
    filtered_image_roidb_train{end+1} = Struct;
end
image_roidb_train = cat(1, filtered_image_roidb_train{:});

previous_model              = cell(2,1);
previous_model{1}           = models{1}.cur_net_file;
previous_model{2}           = models{2}.cur_net_file;

mkdir_if_missing(['output/', 'weakly_cachedir/', opts.cache_name]);
cache_dir       = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, imdbs_name);
debug_cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug');

pre_keeps                   = zeros(numel(image_roidb_train), numel(models));
gamma                       = 0.3;
base_select                 =  [40, 12, 10, 20, 20, 10, 50, 25, 15, 10,...
                                20, 15, 15, 30, 15, 15, 15, 20, 40, 35];
base_select                 = ceil(base_select * (7000 / sum(base_select)));

cache_dir                   = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, [imdbs_name, '_step1']);
debug_cache_dir             = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug_step1');
[previous_model, pre_keeps] = weakly_dual_train_step(image_roidb_train, models, box_param, base_select, pre_keeps, gamma, class_limit, rng_seed, cache_dir, debug_cache_dir);

test_time                   = tic;
fprintf('-------------------- TESTING STEP 1 --------------------\n');
step1_models                = previous_model;

loc_dataset                 = Dataset.voc2012_trainval_ss(dataset, 'train', false, false);
corloc                      = weakly_test_Cor_v2({models{1}.conf, models{2}.conf}, loc_dataset.imdb_train{1}, loc_dataset.roidb_train{1}, ... 
                                'net_defs',        {models{1}.test_net_def_file, models{2}.test_net_def_file}, ... 
                                'net_models',      step1_models, ... 
                                'cache_name',      opts.cache_name, ... 
                                'log_prefix',      'co_cor_final', ...
                                'ignore_cache',    true);

res_path                    = weakly_online_test({models{1}.conf, models{2}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{1}.test_net_def_file, models{2}.test_net_def_file}, ...
                                'net_models',       step1_models, ...
                                'log_prefix',       'co_final', ...
                                'cache_name',       opts.cache_name,...
                                'ignore_cache',     true);

final_path                  = prepare_submit_voc(res_path, opts.cache_name, 'stage1_submit', 0.0001);

%{
base_select            = 4300;
models{1}.cur_net_file = previous_model{1};
models{2}.cur_net_file = previous_model{2};
cache_dir              = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, [imdbs_name, '_step2']);
debug_cache_dir        = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug_step2');
[previous_model, pre_keeps] = weakly_dual_train_step(image_roidb_train, models, box_param, base_select, pre_keeps, gamma, rng_seed, cache_dir, debug_cache_dir);

%}
