% pascal_sentences-features 
clear; clc;
% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_dir = fullfile(pwd, 'results', 'voc07_cnn');
mkdir_if_missing(log_dir);
system(sprintf('rm -rf %s/*',log_dir));
log_file = fullfile(log_dir, ['Examples_VOC07_ALM_', timestamp, '.txt']);
diary(log_file);
load ('VOC07-ALL.mat');
train_features = []; test_features = [];
train_label = Train.label;
test_label = Test.label;
%fea_names = {'vgg16', 'vgg19', 'res152', 'res101', 'res50', 'alex', 'caffe', 'google'};
fea_names = {'vgg16', 'vgg19', 'res101', 'res50', 'alex', 'caffe', 'google'};
%fea_names = {'vgg16', 'vgg19', 'res152', 'res101', 'res50', 'google'};
for index = 1:length(fea_names)
    train_features{end+1} = eval(['Train.', fea_names{index}]);
    test_features{end+1} = eval(['Test.', fea_names{index}]);
end
[train_label, order] = sort(train_label);

for index = 1:numel(test_features)
   test_features{index}  = sparse(test_features{index});
   train_features{index} = sparse(train_features{index});
   train_features{index} = train_features{index}(order, :);
end
results  = zeros(numel(test_label), numel(test_features));
accs     = zeros(numel(test_features), 1);
ForMerge = cell(numel(test_features), 1);
train_time = cell(numel(test_features), 1);
for index = 1:numel(test_features)
    train_data = train_features{index};
    test_data  = test_features{index};
    tic;
    model = train(train_label, train_data);
    train_time{index} = toc;
    [predict_label, accuracy, dec_values] = predict(test_label, test_data, model);
    results(:, index) = predict_label;
    ForMerge{index} = dec_values;
    [MX, idx] = max(dec_values, [], 2);
    assert(all(predict_label == idx)); 
    accs(index) = accuracy(1);
end
[MergeRes] = MergeResults(ForMerge, [], 1);
[~, MergeRes] = max(MergeRes, [], 2);
fprintf('Total classifier : %d, mean accuracy : %.4f, Merge accuracy : %.4f\n', numel(accs), mean(accs), sum(MergeRes==test_label) / numel(test_label));
for index = 1:numel(test_features)
    fprintf('The %2dth classifier accuracy : %.3f\n', index, sum(results(:,index)==test_label)/numel(test_label));
end
K = numel(unique(train_label));
%clearvars -except results test_label K train_label ForMerge log_dir accs MergeRes;
opts = default_alm();
opts.K = K;
opts.max_iters = 300;
opts.p = 1.01; opts.mu = 10;
options = default_opts();
options.maxit = 200;
options.rel_tol_change_res = 1e-6;
opts.options = options;

[X, param] = SolverALM(results, test_label, opts);
fprintf('Total classifier : %d, mean accuracy : %.4f, Merge accuracy : %.4f\n', numel(accs), mean(accs), sum(MergeRes==test_label) / numel(test_label));

save(fullfile(log_dir, 'VOC07-CNN-Results.mat'), '-v7.3');
diary off;
