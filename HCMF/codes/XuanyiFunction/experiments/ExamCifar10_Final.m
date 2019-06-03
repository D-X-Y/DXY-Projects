% pascal_sentences-features 
clear; clc;
% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_dir = fullfile(pwd, 'results', 'cifar10-final');
mkdir_if_missing(log_dir);
%system(sprintf('rm -rf %s/*',log_dir));
log_file = fullfile(log_dir, ['Examples_Cifar10_ALM_', timestamp, '.txt']);
diary(log_file);
load ('Cifar10-All-14.mat');
%load ('Cifar---16.mat');

train_label = double(train_label);
[train_label, order] = sort(train_label);
test_label = double(test_label);
train_label = train_label + 1;
test_label = test_label + 1;
for index = 1:numel(test_features)
   test_features{index} = sparse(test_features{index});
   train_features{index} = sparse(train_features{index});
   train_features{index} = train_features{index}(order, :);
end
M = numel(test_features);
results  = zeros(numel(test_label), M);
accs     = zeros(M, 1);
ForMerge = cell(M, 1);
train_scores   = cell(M, 1);
test_scores    = cell(M, 1);
train_time     = 0;
for index = 1:numel(test_features)
    train_data = train_features{index};
    test_data  = test_features{index};
    tic;
    model = train(train_label, train_data, '-s 2');
    train_time = train_time + toc;
    [predict_label, accuracy, dec_values] = predict(test_label, test_data, model);
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
fprintf('Total classifier : %d, mean accuracy : %.4f, Merge accuracy : %.4f\n', numel(accs), mean(accs), mean(MergeRes==test_label));
for index = 1:numel(test_features)
    fprintf('The %2dth classifier accuracy : %.3f\n', index, sum(results(:,index)==test_label)/numel(test_label));
end
K = numel(unique(train_label));
%clearvars -except results test_label K train_label ForMerge log_dir acc MergeRes accs;
opts = default_alm();
opts.K = K;
opts.max_iters = 200;
opts.p  = 1.05;
opts.mu = 2;
options = default_opts();
options.maxit = 60;
opts.options = options;
%options.rel_tol_change_res = 1e-7;
fuse_time = tic;
[X, param] = SolverYan(results, test_label, opts);
fuse_time = toc(fuse_time);
[SCORES, Tlabel] = GetTrueLabel(X, param.M, param.K);
Our_acc    = mean(Tlabel == test_label);
opts.lamda = 0.1;
opts.K = K;
opts.max_iters = 10000;
opts.gamma = 0.01;
opts.beta  = 0.01;
rcec_time = tic;
[XX, param] = SolverRCEC(results, test_label, opts);
rcec_time = toc(rcec_time);
[SCORES, Xlabel] = GetTrueLabel(XX, param.M, param.K);
RCEC_acc   = mean(Xlabel==test_label);

%[mkl_acc, devs] = SolverMKL(train_features, train_label, test_features, test_label, K);
mkl_acc = -1;
%lpboost_acc = SolverLPB(train_scores, train_label, test_scores, test_label, 1);
lpboost_acc  = -1;
FWOT_acc    = -1;
fprintf('Merge_acc : %.4f, RCEC_acc : %.4f, FWOT : %.4f, MKL : %.4f, LPBoost : %.4f, Our : %.4f\n', Merge_acc, RCEC_acc, FWOT_acc, mkl_acc, lpboost_acc, Our_acc);
Strategy_Compare(X, test_scores, test_label, param.M, K)

save(fullfile(log_dir, 'cifar10.mat'), '-v7.3');
diary off;
