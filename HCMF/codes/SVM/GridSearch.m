function [best_C, best_gamma] = GridSearch(TrainMatrix, trainLabel, folds)
    tic;
    %# grid of parameters
    [C,gamma] = meshgrid(-5:2:15, -15:2:3);

    %# grid search, and cross-validation
    cv_acc = zeros(numel(C),1);
    for i=1:numel(C)
        cv_acc(i) = svmtrain(trainLabel, TrainMatrix, ...
                    sprintf('-t 4 -c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
    end

    %# pair (C,gamma) with best accuracy
    [~,idx] = max(cv_acc);
    best_C = 2^C(idx);
    best_gamma = 2^gamma(idx);
    fprintf('Cross-Validation Accuracy : %.2f, Best C : %.3f, Best Gamma : %.4f, Cost %.1f s\n', cv_acc(idx), best_C, best_gamma, toc);
end
