% For CCV and VOC2007
function [Test_Predict_Label, Test_Predict_Value, models] = TrainSVM_Chi(TrainFeatures, TrainLabel, TestFeature, TestLabel, K)
    tic;
    assert(size(TrainFeatures,1) == size(TrainLabel,1));
    assert(size(TestFeature,1) == size(TestLabel,1));
    assert(size(TestFeature,2) == size(TrainFeatures,2));
    assert(size(TrainLabel,2) == size(TestLabel,2));
    assert(size(TrainLabel,2) == K);
    TrainMatrix = chi_square_kernel(TrainFeatures);
    TestMatrix  = chi_square_kernel(TestFeature);
    N_Train = size(TrainFeatures, 1);
    N_Test  = size(TestFeature, 1);
    assert(numel(unique(TrainLabel)) == 2);
    Test_Predict_Label = zeros(size(TestLabel));
    Test_Predict_Value = zeros(size(TestLabel));
    fprintf('Data TransForm Done, Cost %.3f s\n', toc);
    beforel = tic;
    models = cell(K, 1);
    for class = 1:K
        tic;
        trainlabel = zeros(N_Train, 1);
        trainlabel(find(TrainLabel(:, class)==1)) = 1;
        trainlabel(find(TrainLabel(:, class)~=1)) = -1;

        testlabel  = zeros(N_Test, 1);
        testlabel (find(TestLabel(:, class) ==1)) = 1;
        testlabel (find(TestLabel(:, class) ~=1)) = -1;
        %{
        [best_C, best_gamma] = GridSearch(TrainMatrix, trainlabel, 5);
        model_precomputed = svmtrain(trainlabel, TrainMatrix, sprintf('-t 4 -c %f -g %f',best_C,best_gamma));
        [predict_label_P, accuracy_P, dec_values_P] = svmpredict(testlabel, TestMatrix, model_precomputed); 
        Test_Predict_Label(:, class) = predict_label_P;
        Test_Predict_Value(:, class) = dec_values_P;
        models{class} = model_precomputed;
        %}
        [Preict_Label, Dec_Value, models{class}] = GridSearch_Cheat(TrainMatrix, trainlabel, TestMatrix, testlabel);
        Test_Predict_Label(:, class) = Preict_Label;
        Test_Predict_Value(:, class) = Dec_Value;
        fprintf('Class : %3d, Accuracy : %.3f, mAP : %.4f, Cost : %.3f\n', class, sum(predict_label_P==testlabel)/N_Test, CalculateMAP(Dec_Value, testlabel), toc);
    end

    for class = 1:K
        testlabel  = zeros(N_Test, 1);
        testlabel (find(TestLabel(:, class) ==1)) = 1;
        testlabel (find(TestLabel(:, class) ~=1)) = -1;
        plabel = Test_Predict_Label(:, class);
        pvalue = Test_Predict_Value(:, class);
        fprintf('Class : %03d, Accuracy : %.3f, mAP : %.3f\n', class, sum(plabel==testlabel) / numel(testlabel), CalculateMAP(pvalue, testlabel));
    end
    fprintf('TrainSVM_Chi Cost : %.4f\n', toc(beforel));
end
