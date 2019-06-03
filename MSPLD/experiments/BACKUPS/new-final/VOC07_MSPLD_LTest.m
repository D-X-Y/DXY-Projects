figure('visible', 'off');
clear; close all;
clear mex;
clear is_valid_handle; % to clear init_key
%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% global parameters
extra_para                  = load(fullfile(pwd, 'models', 'pre_trained_models', 'fast_box_param.mat'));
classes                     = extra_para.VOCopts.classes;
rng_seed                    = 5;
per_class_sample            = 3;
print_result                = false;
use_flipped                 = true;
gamma                       = 0.2;
base_select                 = [3, 4, 5, 6];
% model
models                      = cell(3,1);
box_param                   = cell(3,1);
models{1}.solver_def_file   = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'resnet50_res5', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'resnet50_res5', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
models{1}.cur_net_file      = 'unset';
models{1}.name              = 'Fast-ResNet50-SIMPLE';
models{1}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image.mat');
models{1}.conf              = fast_rcnn_config('image_means', models{1}.mean_image, ...
                                               'classes', classes, ...
                                               'max_epoch', 9, 'step_epoch', 8, ...
                                               'regression', true, 'max_rois_num_in_gpu', 1300);
box_param{1}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'fast_box_param.mat'));

models{2}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-101L_res3a', 'solver_lr1_3.prototxt');
models{2}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-101L_res3a', 'test.prototxt');
models{2}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-101L', 'ResNet-101-model.caffemodel');
models{2}.cur_net_file      = 'unset';
models{2}.name              = 'Rfcn-ResNet101-SIMPLE';
models{2}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-101L', 'mean_image.mat');
models{2}.conf              = rfcn_config_simple('image_means', models{2}.mean_image, ...
                                               'classes', classes, ...
                                               'max_epoch', 9, 'step_epoch', 8, 'regression', true);
box_param{2}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'rfcn_box_param.mat'));

models{3}.solver_def_file   = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'solver_lr1_3.prototxt');
models{3}.test_net_def_file = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'vgg_16layers_conv3_1', 'test.prototxt');
models{3}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'VGG16', 'vgg16.caffemodel');
models{3}.cur_net_file      = 'unset';
models{3}.name              = 'Fast-VGG16-SIMPLE';
models{3}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'VGG16', 'mean_image.mat');
models{3}.conf              = fast_rcnn_config('image_means', models{3}.mean_image, ...
                                                 'classes', extra_para.VOCopts.classes, ...
                                                 'max_epoch', 9, 'step_epoch', 8, 'regression', true);
box_param{3}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'fast_box_param.mat'));

% cache name
opts.cache_name             = ['LTest_', models{1}.name, '_', models{2}.name, '_', models{3}.name];
opts.cache_name             = [opts.cache_name, '_per-', num2str(mean(per_class_sample)), '_seed-', num2str(rng_seed), '_base_select-', num2str(mean(base_select))];
% train/test data
fprintf('Loading dataset...');
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', use_flipped);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false);
fprintf('Done.\n');

fprintf('-------------------- TRAINING --------------------\n');
train_time                  = tic;
opts.rfcn_model             = cell(3, 1);
opts.rfcn_model{1}         = fullfile(pwd, 'output', 'weakly_cachedir', 'MSPLD3_Fast-ResNet50-SIMPLE_Rfcn-ResNet101-SIMPLE_Fast-VGG16-SIMPLE_per-3_seed-5_base_select-4.5', 'voc_2007_trainval', 'Fast-ResNet50-SIMPLE_final.caffemodel');
opts.rfcn_model{2}         = fullfile(pwd, 'output', 'weakly_cachedir', 'MSPLD3_Fast-ResNet50-SIMPLE_Rfcn-ResNet101-SIMPLE_Fast-VGG16-SIMPLE_per-3_seed-5_base_select-4.5', 'voc_2007_trainval', 'Rfcn-ResNet101-SIMPLE_final.caffemodel');
opts.rfcn_model{3}         = fullfile(pwd, 'output', 'weakly_cachedir', 'MSPLD3_Fast-ResNet50-SIMPLE_Rfcn-ResNet101-SIMPLE_Fast-VGG16-SIMPLE_per-3_seed-5_base_select-4.5', 'voc_2007_trainval', 'Fast-VGG16-SIMPLE_final.caffemodel');
assert(isfield(opts, 'rfcn_model') ~= 0, 'not found trained model');
train_time                  = toc(train_time);

fprintf('-------------------- TESTING --------------------\n');
assert(numel(opts.rfcn_model) == numel(models));
test_time                   = tic;
net_defs                    = [];
net_models                  = [];
net_confs                   = [];
for idx = 1:numel(models)
    net_defs{idx}           = models{idx}.test_net_def_file;
    net_models{idx}         = opts.rfcn_model{idx};
    net_confs{idx}          = models{idx}.conf;
end
loc_dataset                 = Dataset.voc2007_trainval_ss([], 'train', false);
Corloc                      = weakly_test_Cor_v2(net_confs, loc_dataset.imdb_train{1}, loc_dataset.roidb_train{1}, ...
                                'net_defs',         net_defs, ...
                                'net_models',       net_models, ...
                                'cache_name',       opts.cache_name,...
                                'log_prefix',       ['Triple_final_'], ...
                                'ignore_cache',     true);
mAPs                        = weakly_test_mAP_v2(net_confs, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         net_defs, ...
                                'net_models',       net_models, ...
                                'cache_name',       opts.cache_name,...
                                'log_prefix',       ['Triple_final_'],...
                                'ignore_cache',     true);
test_time                   = toc(test_time);
fprintf('Training Cost : %.1f s, Test Cost : %.1f s, mAP : %.2f, Corloc : %.2f\n', train_time, test_time, mAPs, Corloc);
