% For CCV and VOC2007
function TTTemp(TrainFeatures, TrainLabel, TestFeature, TestLabel, K)
    tic;
    assert(size(TrainFeatures,1) == size(TrainLabel,1));
    assert(size(TestFeature,1) == size(TestLabel,1));
    assert(size(TestFeature,2) == size(TrainFeatures,2));
    assert(size(TrainLabel,2) == size(TestLabel,2));
    assert(size(TrainLabel,2) == K);
    TrainMatrix = chi_square_kernel(TrainFeatures, TrainFeatures);
    TestMatrix  = chi_square_kernel(TestFeature, TrainFeatures);
    N_Train = size(TrainFeatures, 1);
    N_Test  = size(TestFeature, 1);
    assert(numel(unique(TrainLabel)) == 2);
    Test_Predict_Label = zeros(size(TestLabel));
    Test_Predict_Value = zeros(size(TestLabel));
    fprintf('Data TransForm Done, Cost %.3f s\n', toc);
    APs = zeros(K,1);
    for class = 1:K
        trainlabel = zeros(N_Train, 1);
        trainlabel(find(TrainLabel(:, class)==1)) = 1;
        trainlabel(find(TrainLabel(:, class)~=1)) = -1;

        testlabel  = zeros(N_Test, 1);
        testlabel (find(TestLabel(:, class) ==1)) = 1;
        testlabel (find(TestLabel(:, class) ~=1)) = -1;
        model = svmtrain(trainlabel, TrainMatrix, sprintf('-t 4'));
        [PPLabel, ~, DDValue] = svmpredict(testlabel, TestMatrix, model);
        accuracy = sum(PPLabel == testlabel) / numel(testlabel);
        APs(class) = CalculateMAP(DDValue, testlabel);
        fprintf('::: Accuracy : %.3f, mAP : %.4f\n', accuracy, APs(class));
    end
    fprintf('Mean AP : %.4f\n', mean(APs));

end
