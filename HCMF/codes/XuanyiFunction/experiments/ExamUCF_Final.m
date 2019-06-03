clear; clc;
% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_dir = fullfile(pwd, 'results', 'ucf101_P');
mkdir_if_missing(log_dir);
system(sprintf('rm -rf %s/*',log_dir));
log_file = fullfile(log_dir, ['Examples_UCF_ALM_', timestamp, '.txt']);
diary(log_file);
load('UCF101-All-Features.mat');
[train_label, order] = sort(train_label);
for index = 1:numel(test_features)
   train_features{index} = train_features{index}(order, :);
end
results = zeros(numel(test_label), numel(test_features));
ForMerge = [];
M = numel(test_features);
train_scores   = cell(M, 1);
test_scores    = cell(M, 1);
for index = 1:numel(train_features)
    if ( index <= 3 ) 
        ForMerge{index} = test_features{index};
        [MX, idx] = max(test_features{index}, [], 2);
        accs(index) = mean(idx == test_label);
        train_scores{index} = train_features{index};
        test_scores{index} = test_features{index};
        results(:, index) = idx;
    else
        train_data = sparse(train_features{index});
        test_data  = sparse(test_features{index});
        model = train(train_label, train_data);
        [predict_label, accuracy, dec_values] = predict(test_label, test_data, model);
        results(:, index) = predict_label;
        ForMerge{index} = dec_values;
        [MX, idx] = max(dec_values, [], 2);
        assert(all(predict_label == idx));
        accs(index) = accuracy(1) / 100;
        test_scores{index}  = dec_values;
        [predict_label, accuracy, dec_values] = predict(train_label, sparse(train_data), model);
        train_scores{index} = dec_values;
    end
end
K = 101;
%CompareAlg(train_features, train_label, test_features, test_label, ForMerge, K);
[MergeRes]    = MergeResults(ForMerge, [], 2);
[~, MergeRes] = max(MergeRes, [], 2);
MergeAcc      = mean(MergeRes==test_label);
fprintf('Total classifier : %d, Merge accuracy : %.4f\n', numel(accs), mean(MergeRes==test_label));

opts = default_alm();
opts.K = K;
opts.max_iters = 10;
opts.p = 1.01;
opts.mu = 1;
options = default_opts();
options.maxit = 100;
opts.options = options;
opts.options.rel_tol_change_res = 1e-8;
disp(opts.options);
[X, param] = SolverALM(results, test_label, opts);
[SCORES, Tlabel] = GetTrueLabel(X, param.M, param.K);
OurAcc     = mean(Tlabel == test_label);

opts.lamda = 0.1;
opts.max_iters = 50;
opts.gamma = 0.01;
opts.beta  = 0.01;
[XX, param] = SolverRCEC(results, test_label, opts);
[SCORES, Xlabel] = GetTrueLabel(XX, param.M, param.K);
RCEC_acc   = mean(Xlabel==test_label);

[mkl_acc, devs] = SolverMKL(train_features, train_label, test_features, test_label, K);
lpboost_acc = SolverLPB(train_scores, train_label, test_scores, test_label, 1);

FWOT_acc    = SolverFWOT(train_scores, train_label, test_scores, test_label, K, 1);

for index = 1:numel(accs)
    fprintf('The %d th accuracy : %.4f\n', index, accs(index));
end
fprintf('Merge : %.4f  ,  OurORLF : %.4f\n', MergeAcc, OurAcc);
fprintf('Merge_acc : %.4f, RCEC_acc : %.4f, FWOT : %.4f, MKL : %.4f, LPBoost : %.4f, Our : %.4f\n', Merge_acc, RCEC_acc, FWOT_acc, mkl_acc, lpboost_acc, Our_acc);

%save(fullfile(log_dir, 'UCF101-Results.mat'), '-v7.3');
diary off;
