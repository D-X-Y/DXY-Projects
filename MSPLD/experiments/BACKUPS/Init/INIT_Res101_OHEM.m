figure('visible', 'off');
clear; close all; clc;
clear mex;
clear is_valid_handle; % to clear init_key
%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% global parameters     mAP = [ 30.3223 ]
extra_para                  = load(fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat'));
rng_seed                    = 5;
per_class_sample            = 3;
print_result                = true;
use_flipped                 = true;
gamma                       = 0.0;
base_select                 = zeros(0,1);
% model
models                      = cell(1,1);
models{1}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-101L_OHEM_res3a', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-101L_OHEM_res3a', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-101L', 'ResNet-101-model.caffemodel');
models{1}.cur_net_file      = 'unset';
models{1}.name              = 'ResNet101-OHEM';
models{1}.mean_image        = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-101L', 'mean_image.mat');
models{1}.conf              = rfcn_config_ohem('image_means', models{1}.mean_image, ...
                                               'classes', extra_para.VOCopts.classes, ...
                                               'max_epoch', 8, 'step_epoch', 7, 'regression', true);

% cache name
opts.cache_name             = ['INIT_', models{1}.name];
box_param.bbox_means        = extra_para.bbox_means;
box_param.bbox_stds         = extra_para.bbox_stds;
opts.cache_name             = [opts.cache_name, '_per-', num2str(mean(per_class_sample)), '_seed-', num2str(rng_seed)];
% train/test data
fprintf('Loading dataset...');
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', use_flipped);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false);
fprintf('Done.\n');

fprintf('-------------------- TRAINING --------------------\n');
train_time                  = tic;
opts.rfcn_model             = weakly_co_train_final(dataset.imdb_train, dataset.roidb_train, models, ...
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
mAPs                        = weakly_co_test_mAP({models{1}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{1}.test_net_def_file}, ...
                                'net_models',       {opts.rfcn_model{1}}, ...
                                'cache_name',       opts.cache_name, ...
                                'log_prefix',       [models{1}.name, '_final_'], ...
                                'rng_seed',         rng_seed, ...
                                'ignore_cache',     true);
loc_dataset                 = Dataset.voc2007_trainval_ss([], 'train', false);
Corloc                      = weakly_co_test_Cor({models{1}.conf}, loc_dataset.imdb_train{1}, loc_dataset.roidb_train{1}, ...
                                'net_defs',         {models{1}.test_net_def_file}, ...
                                'net_models',       {opts.rfcn_model{1}}, ...
                                'cache_name',       opts.cache_name, ...
                                'rng_seed',         rng_seed, ...
                                'ignore_cache',     true);
fprintf('Training Cost : %.1f s, Test Cost : %.1f s, mAP : %.2f, Corloc : %.2f\n', train_time, test_time, mAPs, Corloc);

fprintf('----------------------------------All Test-----------------------------\n');
imdbs_name          = cell2mat(cellfun(@(x) x.name, dataset.imdb_train,'UniformOutput', false));
all_test_time       = tic;
rfcn_model          = cell(models{1}.conf.max_epoch, 1);
for idx = 1:models{1}.conf.max_epoch
    rfcn_model{idx} = fullfile(pwd, 'output', 'weakly_cachedir' , opts.cache_name, imdbs_name, [models{1}.name, '_Loop_0_epoch_', num2str(idx), '.caffemodel']);
	assert(exist(rfcn_model{idx}, 'file') ~= 0, 'not found trained model');
end

allAPs = zeros(models{1}.conf.max_epoch, 1);
for idx = 1:models{1}.conf.max_epoch
allAPs(idx)                 = weakly_co_test_mAP({models{1}.conf}, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{1}.test_net_def_file}, ...
                                'net_models',       {rfcn_model{idx}}, ...
                                'cache_name',       opts.cache_name, ...
                                'log_prefix',       [models{1}.name, '_epoch_', num2str(idx)], ...
                                'rng_seed',         rng_seed, ...
                                'ignore_cache',     true);

end
