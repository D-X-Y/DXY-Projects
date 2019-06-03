function [acc, predicted_results] = SolverFWOT(train_scores, train_label, test_scores, test_label, K, C)
    simple_path{1} = '/home/dongxuanyi/AAAI/MKL/';
    simple_path{2} = '/home/dongxuanyi/AAAI/MKL/simplemkl/';
    addpath(simple_path{1}); addpath(simple_path{2});
    being_time = tic;
    [acc, predicted_results] = FWOT(train_scores, train_label, test_scores, test_label, K, C);
    fprintf('SolverFWOT : : : : %.3f s\n', toc(being_time));
    rmpath(simple_path{1}); rmpath(simple_path{2});
end

