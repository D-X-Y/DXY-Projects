% train_data_name, test_data_name
% ExamEnsemble('dna.scale.tr', 'dna.scale.t', '-s 1');
% ExamEnsemble('letter.scale.tr', 'letter.scale.t', '-s 1');
% ExamEnsemble('mnist.scale', 'mnist.scale.t');
% ExamEnsemble('news20.scale', 'news20.t.scale', '-s 1');
% ExamEnsemble('pendigits', 'pendigits.t', '-s 1');
% ExamEnsemble('poker', 'poker.t', '-s 1');
% ExamEnsemble('shuttle.scale.tr', 'shuttle.scale.t', '-s 1');
% ExamEnsemble('satimage.scale.tr', 'satimage.scale.t', '-s 1');
% ExamEnsemble(sector.scale, sector.t.scale);
function ExamEnsemble(train_data_name, test_data_name, train_opts)
svm_data_path = '/home/dongxuanyi/AAAI/libsvm-data/';
%addpath(svm_data_path);
[train_label, train_data] = libsvmread(fullfile(svm_data_path, train_data_name));
[test_label, test_data] = libsvmread(fullfile(svm_data_path, test_data_name));
test_label  = SPARSE(test_label);
train_label = SPARSE(train_label);
%rmpath(svm_data_path);
% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_dir = fullfile(pwd, 'log');
mkdir_if_missing(log_dir);
log_file = fullfile(log_dir, ['Examples_Ensemble_', train_data_name, '_ALM_', timestamp, '.txt']);
diary(log_file);
fprintf('Total Class : %d,  Train Instances : %d, Feature Dims : %d\n', numel(unique(train_label)), size(train_data,1), size(train_data,2));
fprintf('Total Class : %d,  Test  Instances : %d, Feature Dims : %d\n', numel(unique(test_label)), size(test_label,1), size(train_data,2));
sample_ratio = 0.5;
sample_iters = 10;
train_data = sparse(train_data);
test_data = sparse(test_data);
% First, Check Total Train
model = train(train_label, train_data, train_opts);
[predicted_label, accuracy, prob_estimates] = predict(test_label, test_data, model);
original_accuracy = accuracy(1);
accs = zeros(sample_iters, 1);
results = cell(sample_iters, 1);
K = numel(unique(train_label));
train_features = cell(sample_iters, 1);
test_features = cell(sample_iters, 1);
train_scores = cell(sample_iters, 1);
test_scores = cell(sample_iters, 1);

for index = 1:sample_iters
    [subset, ~] = split_label(train_label, sample_ratio);
    sub_train_data  = train_data(subset, :);
    sub_train_label = train_label(subset, :);
    train_features{index} = sub_train_data;
    test_features{index}  = test_data;
    [~, order] = sort(sub_train_label);
    sub_train_data = sub_train_data(order, :);
    sub_train_label = sub_train_label(order, :);
    model = train(sub_train_label, sub_train_data, train_opts);
    [predicted_label, accuracy, prob_estimates] = predict(test_label, test_data, model);
    accs(index) = accuracy(1);
    assert( K == size(prob_estimates,2));
    results{index} = prob_estimates;
    [mx, idx] = max(prob_estimates, [], 2);
    assert(all(predicted_label == idx));
    test_scores{index} = prob_estimates;
    [predicted_label, accuracy, prob_estimates] = predict(train_label, train_data, model);
    train_scores{index} = prob_estimates;

end
fprintf('Total classifier : %d, for all train accuracy : %.4f, mean accuracy : %.4f\n', numel(results), original_accuracy, mean(accs));
Temp = zeros(numel(test_label), sample_iters);
for index = 1:sample_iters
    [mx, idx] = max(results{index}, [], 2);
    fprintf('The %2dth classifier accuracy : %.3f, %.3f\n', index, accs(index), sum(idx==test_label)/numel(test_label));
    Temp(:, index) = idx;
end
[MergeRes] = MergeResults(results, [], 2);
[MX, IDX]  = max(MergeRes, [], 2);
Merge_acc  = mean(IDX == test_label);
fprintf('Merge Result Accuracy : %.3f \n',Merge_acc);
results = Temp;
%clearvars -except results test_label K;
opts = default_alm();
opts.K = K;
opts.max_iters = 60;
opts.p = 1.05; 
opts.mu = 5;
options = default_opts();   options.maxit = 200;
opts.options = options;
[X, param] = SolverALM(results, test_label, opts);
[SCORES, Tlabel] = GetTrueLabel(X, param.M, param.K);
Our_acc   = mean(Tlabel==test_label);
fprintf('[Merge accuracy : %.4f] vs [Final X : %.4f]\n', Merge_acc, Our_acc);
opts.lamda = 0.1;
opts.K = K;
opts.max_iters = 20;
opts.gamma = 0.01;
opts.beta  = 0.01;
[XX, param] = SolverRCEC(results, test_label, opts);
[SCORES, Xlabel] = GetTrueLabel(XX, param.M, param.K);
RCEC_acc   = mean(Xlabel==test_label);
for index = 1:size(results,2)
    fprintf('The %2dth classifier accuracy : %.3f\n', index, mean(results(:,index)==test_label));
end
fprintf('Merge : %.4f vs OUR : %.4f vs [RCEC : %.4f]\n', Merge_acc, Our_acc, RCEC_acc);
lpboost_acc = SolverLPB(train_scores, train_label, test_scores, test_label, 1);
fprintf('Merge_acc : %.4f, RCEC_acc : %.4f, FWOT : %.4f, MKL : %.4f, LPBoost : %.4f, Our : %.4f\n', Merge_acc, RCEC_acc, -1, -1, lpboost_acc, Our_acc);

diary off;

end

function label = SPARSE(label)
    ALL = unique(label);
    Temp = zeros(size(label));
    for i = 1:numel(ALL)
        Temp(find(label==ALL(i))) = i;
    end
    label = Temp;
end
