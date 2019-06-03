clear; clc;
% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_dir = fullfile(pwd, 'results', 'VOC_Final');
mkdir_if_missing(log_dir);
%system(sprintf('rm -rf %s/*',log_dir));
log_file = fullfile(log_dir, ['Examples_VOC_Final_ALM_', timestamp, '.txt']);
diary(log_file);
load ('VOC07-ALL.mat');
parpool(16);
fea_names = {'vgg16', 'vgg19', 'res152', 'res101', 'res50', 'alex', 'caffe', 'google'};
%fea_names = {'vgg16', 'vgg19', 'res101', 'res50', 'alex', 'caffe', 'google'};
%select = [1,2,4,5,6,7,8];
%select = [1,2,3,4,5,6,7,8];
features = [];
label    = [Train.label; Test.label];
for index = 1:length(fea_names)
    fprintf('Select %s\n', fea_names{index});
    features{end+1} = [eval(['Train.', fea_names{index}]); eval(['Test.', fea_names{index}])];
end
ratio = 0.5;
total_number = numel(label);
train_id    = randperm(total_number, ceil(ratio*total_number))';
test_id     = setdiff((1:total_number)', train_id);
train_label = label(train_id);
[train_label, order] = sort(train_label);
train_id    = train_id(order, :);
test_label  = label(test_id);


results  = zeros(numel(test_label), numel(features));
accs     = zeros(numel(features), 1);
ForMerge = cell(numel(features), 1);
train_scores   = cell(numel(features), 1);
test_scores    = cell(numel(features), 1);
train_features = cell(numel(features), 1);
test_features  = cell(numel(features), 1);
for index = 1:numel(features)
    feature_data = features{index};
    train_data = feature_data(train_id, :);
    test_data  = feature_data(test_id, :);
    train_features{index} = train_data;
    test_features{index}  = test_data;
    model = train(train_label, sparse(train_data));
    [predict_label, accuracy, dec_values] = predict(test_label, sparse(test_data), model);
    results(:, index) = predict_label;
    ForMerge{index} = dec_values;
    [MX, idx] = max(dec_values, [], 2);
    assert(all(predict_label == idx));
    accs(index) = accuracy(1);

    %% For LPBoost
    test_scores{index}  = dec_values;
    [predict_label, accuracy, dec_values] = predict(train_label, sparse(train_data), model);
    train_scores{index} = dec_values;
end
M = numel(features);

[MergeRes] = MergeResults(ForMerge, [], 1);
[~, MergeRes] = max(MergeRes, [], 2);
fprintf('Total classifier : %d, mean accuracy : %.4f, Merge accuracy : %.4f\n', numel(accs), mean(accs), sum(MergeRes==test_label) / numel(test_label));
for index = 1:M
    fprintf('The %2dth classifier accuracy : %.3f\n', index, mean(results(:,index)==test_label));
end
%clearvars -except results test_label K MergeRes ForMerge test_id;
K = numel(unique(label));
opts = default_alm();
opts.K = K;
opts.max_iters = 100;
opts.p = 1.01;
opts.mu = 2;
options = default_opts();
options.maxit = 200;
opts.options = options;
opts.options.rel_tol_change_res = 1e-8;
disp(opts.options);
[X, OR_param] = SolverYan(results, test_label, opts);
[SCORES, Tlabel] = GetTrueLabel(X, M, K);
Merge_acc = mean(MergeRes==test_label);
Our_acc   = mean(Tlabel==test_label);
fprintf('[Merge accuracy : %.4f] vs [Final X : %.4f]\n', Merge_acc, Our_acc);

opts.lamda = 0.1;
opts.K = K;
opts.max_iters = 10000;
opts.gamma = 0.01;
opts.beta  = 0.01;
[XX, RC_param] = SolverRCEC(results, test_label, opts);
[SCORES, Xlabel] = GetTrueLabel(XX, M, K);

RCEC_acc   = mean(Xlabel==test_label);
for index = 1:M
    fprintf('The %2dth classifier accuracy : %.3f\n', index, mean(results(:,index)==test_label));
end
fprintf('[Merge accuracy : %.4f] vs [OUR : %.4f] vs [RCEC : %.4f]\n', mean(MergeRes==test_label), mean(Tlabel==test_label), RCEC_acc);

[mkl_acc, devs] = SolverMKL(train_features, train_label, test_features, test_label, K);
lpboost_acc = SolverLPB(train_scores, train_label, test_scores, test_label, 1);
FWOT_acc    = SolverFWOT(train_scores, train_label, test_scores, test_label, K, 1);
fprintf('Merge_acc : %.4f, RCEC_acc : %.4f, FWOT : %.4f, MKL : %.4f, LPBoost : %.4f, Our : %.4f\n', Merge_acc, RCEC_acc, FWOT_acc, mkl_acc, lpboost_acc, Our_acc);

diary off;
