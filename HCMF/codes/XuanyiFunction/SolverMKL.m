function [acc, devs] = SolverMKL(train_features, train_label, test_features, test_label,  K)
    mkl_path = '/home/dongxuanyi/AAAI/Algorithm/MKL/liblinear-mkl/src/matlab';
    begin_T = tic;
    accuracy = [];  M = numel(train_features);
    train_num = numel(train_label);
    test_num  = numel(test_label);
    assert(numel(test_features) == M);
    assert(numel(unique(train_label))==K);
    assert(numel(unique(test_label))==K);
    % Make train_label from 1->n
    [train_label, order] = sort(train_label);
    for index = 1:M
        train_features{index} = train_features{index}(order,:);
    end
    addpath (mkl_path);
    fprintf('Early fusion, fuse feature, MKL : \n');
    %[predicted_label] = wrapSimpleMKL(train_features, train_label, test_features, K);
    dims = zeros(M, 1);
    for index = 1:M
        assert(size(train_features{index}, 2) == size(test_features{index}, 2));
        dims(index) = size(train_features{index}, 2);
    end
    RR = cumsum(dims);
    LL = RR - dims + 1;
    connect_train_features = zeros(train_num, RR(end));
    connect_test_features  = zeros(test_num, RR(end));
    for index = 1:M
        func{index} = inline(sprintf('x(:,%d:%d)',LL(index), RR(index)));
        connect_train_features(:,LL(index):RR(index)) = train_features{index};
        connect_test_features(:,LL(index):RR(index))  = test_features{index};
    end
    connect_train_features = sparse(connect_train_features);
    connect_test_features  = sparse(connect_test_features);
    mklmodel = train_mkl(train_label, connect_train_features, func, '-s 3');
    [py, acc, devs] = predict_mkl(test_label, connect_test_features, mklmodel);
    fprintf('MKL (multiple features, early fusion) Accuracy : %.4f, cost : %.2f s\n', acc, toc(begin_T));
    rmpath(mkl_path);
end
