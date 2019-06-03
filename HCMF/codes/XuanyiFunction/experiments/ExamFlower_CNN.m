clearvars -except FINAL; clc;
% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_dir = fullfile(pwd, 'results', 'Flower17_CNN');
mkdir_if_missing(log_dir);
system(sprintf('rm -rf %s/*',log_dir));
log_file = fullfile(log_dir, ['Examples_Flower17_ALM_', timestamp, '.txt']);
diary(log_file);
load ('Oxford_Flower_CNN.mat');
select = (1:8);
fprintf('Split 1 : Select Features :');
disp(select);
features = features(select);
%total_number = numel(label);
%[train_id, test_id] = split_data(total_number, ratio);
%train_id = split1.train_id';  
%test_id  = split1.test_id';
ratio = 0.5;
total = numel(label);
train_id = randperm(total, ceil(total*ratio))';
test_id  = setdiff((1:total)', train_id);
train_label = label(train_id);
[train_label, order] = sort(train_label);
train_id = train_id(order, :);
test_label  = label(test_id);

K = numel(unique(label));
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
    test_features{index} = test_data;

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
[MergeRes] = MergeResults(ForMerge, [], 1);
[~, MergeRes] = max(MergeRes, [], 2);
fprintf('Total classifier : %d, mean accuracy : %.4f, Merge accuracy : %.5f\n', numel(accs), mean(accs), mean(MergeRes==test_label));
for index = 1:numel(features)
    fprintf('The %2dth classifier accuracy : %.3f\n', index, mean(results(:,index)==test_label));
end
%clearvars -except results test_label K MergeRes ForMerge test_id;
opts = default_alm();
opts.K = K;
opts.max_iters = 100;
opts.p = 1.01;
opts.mu = 1;
options = default_opts();
options.maxit = 100;
opts.options = options;
opts.options.rel_tol_change_res = 1e-8;
disp(opts.options);
[X, param] = SolverYan(results, test_label, opts);
[SCORES, Tlabel] = GetTrueLabel(X, param.M, param.K);
Merge_acc = mean(MergeRes==test_label);
Our_acc   = mean(Tlabel==test_label);
fprintf('[Merge accuracy : %.4f] vs [Final X : %.4f]\n', Merge_acc, Our_acc);

opts.lamda = 0.1;
opts.K = K;
opts.max_iters = 10000;
opts.gamma = 0.01;
opts.beta  = 0.01;
[XX, param] = SolverRCEC(results, test_label, opts);
[SCORES, Xlabel] = GetTrueLabel(XX, param.M, param.K);

RCEC_acc   = mean(Xlabel==test_label);
for index = 1:numel(features)
    fprintf('The %2dth classifier accuracy : %.3f\n', index, mean(results(:,index)==test_label));
end
fprintf('[Merge accuracy : %.4f] vs [OUR : %.4f] vs [RCEC : %.4f]\n', mean(MergeRes==test_label), mean(Tlabel==test_label), RCEC_acc);

%[mkl_acc, devs] = SolverMKL(train_features, train_label, test_features, test_label, K);
%lpboost_acc = SolverLPB(train_scores, train_label, test_scores, test_label, 1);
%FWOT_acc    = SolverFWOT(train_scores, train_label, test_scores, test_label, K, 1);
%mkl_acc     = mkl_acc / 100;
%fprintf('Merge_acc : %.4f, RCEC_acc : %.4f, FWOT : %.4f, MKL : %.4f, LPBoost : %.4f, Our : %.4f\n', Merge_acc, RCEC_acc, FWOT_acc, mkl_acc, lpboost_acc, Our_acc);
%FINAL = [FINAL;[Merge_acc, RCEC_acc, FWOT_acc, mkl_acc, lpboost_acc, Our_acc]];

diary off;
