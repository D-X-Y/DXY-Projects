% pascal_sentences-features 
clear; clc;
% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_dir = fullfile(pwd, 'results', 'pascal_sentences');
mkdir_if_missing(log_dir);
system(sprintf('rm -rf %s/*',log_dir));
log_file = fullfile(log_dir, ['Examples_Pascal_ALM_', timestamp, '.txt']);
diary(log_file);
load ('pascal_sentences-8.mat');
ratio = 0.5;
total_number = numel(label);
%[train_id, test_id] = split_data(total_number, ratio);
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
[MergeRes]    = MergeResults(ForMerge, [], 1);
[~, MergeRes] = max(MergeRes, [], 2);
Merge_acc     = mean(MergeRes==test_label);
fprintf('Total classifier : %d, mean accuracy : %.4f, Merge accuracy : %.4f\n', numel(accs), mean(accs), Merge_acc);
for index = 1:numel(features)
    fprintf('The %2dth classifier accuracy : %.4f\n', index, sum(results(:,index)==test_label)/numel(test_label));
end
K = numel(unique(label));
opts = default_alm();
opts.K = K;
opts.max_iters = 200;        opts.p = 1.01;        opts.mu = 5;
options = default_opts();   options.maxit = 200;  options.rel_tol_change_res = 1e-8;
%options.rel_grad_tol = 1e-12;
opts.options = options;
[X, param]  = SolverYan(results, test_label, opts);
[~, Xlabel] = GetTrueLabel(X, param.M, param.K);
Our_acc     = mean(Xlabel==test_label);
fprintf('[Merge accuracy : %.4f] [Final X : %.4f]\n', Merge_acc, Our_acc);
opts.lamda = 0.1;
opts.K = K;
opts.max_iters = 500;
opts.gamma = 0.01;
opts.beta  = 0.01;
[XX, RECEparam] = SolverRCEC(results, test_label, opts);
[SCORES, Xlabel] = GetTrueLabel(XX, param.M, param.K);
RCEC_acc   = mean(Xlabel==test_label);
fprintf('Merge : %.4f, Our : %.4f, RCEC : %.4f \n', Merge_acc, Our_acc, RCEC_acc);
lpboost_acc = SolverLPB(train_scores, train_label, test_scores, test_label, 1);

[mkl_acc, devs] = SolverMKL(train_features, train_label, test_features, test_label, K);
FWOT_acc    = SolverFWOT(train_scores, train_label, test_scores, test_label, K, 1);

fprintf('Merge : %.4f, Our : %.4f, RCEC : %.4f , MKL: %.4f, LPBoost : %.4f, FWOT: %.4f \n', Merge_acc, Our_acc, RCEC_acc, mkl_acc, lpboost_acc, FWOT_acc);

%save(fullfile(log_dir, 'pascal_sentences.mat'), '-v7.3');

diary off;
