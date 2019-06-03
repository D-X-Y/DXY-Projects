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
extra_para                  = load(fullfile(pwd, 'models', 'pre_trained_models', 'rfcn_box_param.mat'));
classes                     = extra_para.VOCopts.classes;
rng_seed                    = 5;
per_class_sample            = inf;
print_result                = false;
use_flipped                 = true;
gamma                       = 0.2;
base_select                 = zeros(1, 0);
% model
models                      = cell(1,1);
box_param                   = cell(1,1);

models{1}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-101L_res3a', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-101L_res3a', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-101L', 'ResNet-101-model.caffemodel');
models{1}.cur_net_file      = 'unset';
models{1}.name              = 'Rfcn-ResNet101-SIMPLE';
models{1}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-101L', 'mean_image.mat');
models{1}.conf              = rfcn_config_simple('image_means', models{1}.mean_image, ...
                                               'classes', classes, ...
                                               'max_epoch', 8, 'step_epoch', 7, 'regression', true);
box_param{1}                = load(fullfile(pwd, 'models', 'pre_trained_models', 'rfcn_box_param.mat'));

% cache name
opts.cache_name             = ['INF-', models{1}.name];
opts.cache_name             = [opts.cache_name, '_seed-', num2str(rng_seed)];
% train/test data
fprintf('Loading dataset...');
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', use_flipped);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false);
fprintf('Done.\n');

fprintf('-------------------- TRAINING --------------------\n');
train_time                  = tic;
opts.rfcn_model             = weakly_co_train_v3(dataset.imdb_train, dataset.roidb_train, models, ...
                                'cache_name',       opts.cache_name, ...
                                'per_class_sample', per_class_sample, ...
                                'base_select',      base_select, ...
                                'rng_seed',         rng_seed, ...
                                'use_flipped',      use_flipped, ...
                                'gamma',            gamma, ...
                                'debug',            print_result, ...
                                'box_param',        box_param);
assert(isfield(opts, 'rfcn_model') ~= 0, 'not found trained model');
train_time                  = toc(train_time);

fprintf('-------------------- TESTING --------------------\n');
assert(numel(opts.rfcn_model) == numel(models));
mAPs                        = cell(numel(models), 1);
test_time                   = tic;
net_defs                    = [];
net_models                  = [];
net_confs                   = [];
for idx = 1:numel(models)
    mAPs{idx}               = weakly_co_test_mAP({models{idx}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{idx}.test_net_def_file}, ...
                                'net_models',       opts.rfcn_model(idx), ...
                                'cache_name',       opts.cache_name,...
                                'log_prefix',       [models{idx}.name, '_final_'],...
                                'ignore_cache',     true);
    net_defs{idx}           = models{idx}.test_net_def_file;
    net_models{idx}         = opts.rfcn_model{idx};
    net_confs{idx}          = models{idx}.conf;
end
mAPs{numel(models)+1}       = weakly_test_mAP_v2(net_confs, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         net_defs, ...
                                'net_models',       net_models, ...
                                'cache_name',       opts.cache_name,...
                                'log_prefix',       ['Triple_final_'],...
                                'ignore_cache',     true);
test_time                   = toc(test_time);
loc_dataset                 = Dataset.voc2007_trainval_ss([], 'train', false);
Corloc                      = weakly_test_Cor_v2(net_confs, loc_dataset.imdb_train{1}, loc_dataset.roidb_train{1}, ...
                                'net_defs',         net_defs, ...
                                'net_models',       net_models, ...
                                'cache_name',       opts.cache_name,...
                                'log_prefix',       ['Triple_final_'], ...
                                'ignore_cache',     true);
for idx = 1:numel(models)
    fprintf('%s mAP : %.3f\n', models{idx}.name, mAPs{idx});
end
fprintf('Training Cost : %.1f s, Test Cost : %.1f s, mAP : %.2f, Corloc : %.2f\n', train_time, test_time, mAPs{end}, Corloc);
