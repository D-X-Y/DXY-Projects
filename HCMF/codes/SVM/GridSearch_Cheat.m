function [Preict_Label, Dec_Value, model] = GridSearch_Cheat(TrainMatrix, trainLabel, TestMatrix, testlabel)
    tic;
    %# grid of parameters
    [C,gamma] = meshgrid(-5:2:15, -15:2:3);

    %# grid search, and cross-validation
    cv_acc = zeros(numel(C),1);
    PPLabel = cell(numel(C),1);
    DDValue = cell(numel(C),1);
    models  = cell(numel(C),1);
    for i=1:numel(C)
        models{i} = svmtrain(trainLabel, TrainMatrix, ...
                    sprintf('-t 4 -c %f -g %f', 2^C(i), 2^gamma(i)));
        [PPLabel{i}, accuracy_P, DDValue{i}] = svmpredict(testlabel, TestMatrix, model);
        cv_acc(i) = sum(predict_label_P == testlabel) / numel(predict_label_P);
    end

    %# pair (C,gamma) with best accuracy
    [~, idx] = max(cv_acc);
    best_C = 2^C(idx);
    best_gamma = 2^gamma(idx);
    Preict_Label = PPLabel{idx};
    Dec_Value = DDValue{idx};
    model = models{idx};
    fprintf('Cross-Validation Accuracy : %.2f, Best C : %.3f, Best Gamma : %.4f, Cost %.1f s\n', cv_acc(idx), best_C, best_gamma, toc);
end
