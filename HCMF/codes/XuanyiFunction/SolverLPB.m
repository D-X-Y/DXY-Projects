function acc = SolverLPB(train_scores, train_label, test_scores, test_label, verbose)
    mosek_path{1} = '/home/dongxuanyi/mosek';
    mosek_path{2} = '/home/dongxuanyi/mosek/7/toolbox/r2013a';
    addpath(mosek_path{1}); addpath(mosek_path{2});
    acc = CrossValidation(train_scores, train_label, test_scores, test_label, verbose);
    rmpath(mosek_path{1}); rmpath(mosek_path{2});
end

