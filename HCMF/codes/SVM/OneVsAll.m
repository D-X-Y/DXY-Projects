function predctedLabel = OneVsAll(trainLabel, trainMatrix, testLabel, testMatrix, NumofClass)
    model = cell(NumofClass,1);  % NumofClass = 3 in your case
    for k = 1:NumofClass
        model{k} = svmtrain(double(trainLabel==k), [(1:numel(trainLabel))', trainMatrix], '-c 1 -t 4 -b 1');
    end
    
    %% calculate the probability of different labels
    
    pr = zeros(numel(testLabel), NumofClass);
    for k = 1:NumofClass
        [T, TT, p] = svmpredict(double(testLabel==k), [(1:numel(testLabel))', testMatrix], model{k}, '-b 1');
        pr(:,k) = p(:, model{k}.Label==1);    %# probability of class==k
    end
            
    %% your label prediction will be the one with highest probability:
            
    [~,predctedLabel] = max(pr,[],2);

end
