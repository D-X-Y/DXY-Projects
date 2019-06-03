function Strategy_Compare(X, test_scores, test_label, M, K, E)
    assert(numel(test_scores) == M);
    assert(size(X, 2) == M*K);
    Soft = zeros(size(X));
    for i = 1:M
        Soft(:, (i-1)*K+1 : i*K) = test_scores{i};
    end
    N = size(X,1);
    SUM = zeros(N, K);
    % Original Strategy , 1
    for i = 1:M
        Temp = X(:, (i-1)*K+1 : i*K);
        Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K);
        SUM = SUM + Temp;
    end
    [mx, label] = max(SUM, [], 2);
    fprintf('Strategy 1[original*] : %.5f\n' , mean(label == test_label));
    % Original Strategy , 2 , vote
    SUM = zeros(N, K);
    for i = 1:M
        Temp = X(:, (i-1)*K+1 : i*K);
        [mx, label] = max(Temp, [], 2);
        for n = 1:N
            SUM(n, label(n)) = SUM(n, label(n)) + 1;
        end
    end
    [mx, label] = max(SUM, [], 2);
    fprintf('Strategy 2[vote*****] : %.5f\n' , mean(label == test_label));
    % Original Strategy , 3
    SUM = zeros(N, K);
    for i = 1:M
        Temp = exp(Soft(:, (i-1)*K+1 : i*K));
        Temp = Temp ./ repmat(sum(Temp, 2), 1, K);
        Ori  = Temp;
        Temp = X(:, (i-1)*K+1 : i*K);
        Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K);

        SUM = SUM + Temp.*Ori;
    end
    [mx, label] = max(SUM, [], 2);
    fprintf('Strategy 3[nexp:norm]  %.5f\n' , mean(label == test_label));
    SUM = zeros(N, K);
    for i = 1:M
        Temp = exp(Soft(:, (i-1)*K+1 : i*K));
        Temp = Temp ./ repmat(sum(Temp, 2), 1, K);
        Ori  = Temp;
        Temp = X(:, (i-1)*K+1 : i*K);
        Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K);

        SUM = SUM + Temp.*Ori;
    end
    [mx, label] = max(SUM, [], 2);
    fprintf('Strategy 4[Dexp:prod] : %.5f\n' , mean(label == test_label));
    % Original Strategy , 4
    %{
    SUM = zeros(N, K);
    for i = 1:M
        Temp = Soft(:, (i-1)*K+1 : i*K);
        Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K) / M;
        Ori  = Temp;
        Temp = X(:, (i-1)*K+1 : i*K);
        Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K) / M;
        SUM = SUM + Temp + Ori;
    end
    [mx, label] = max(SUM, [], 2);
    fprintf('Strategy 4[norm:sum] : %.5f\n' , mean(label == test_label));
    % Original Strategy , 5
    SUM = zeros(N, K);
    for i = 1:M
        Temp = exp(Soft(:, (i-1)*K+1 : i*K));
        %Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K) / M;
        Temp = Temp ./ repmat(sum(Temp, 2), 1, K) / M;
        Ori  = Temp;
        Temp = X(:, (i-1)*K+1 : i*K);
        Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K) / M;
        SUM = SUM + Temp + Ori;
    end
    [mx, label] = max(SUM, [], 2);
    fprintf('Strategy 5[exp:sum] : %.5f\n' , mean(label == test_label));
    % Original Strategy , 6
    SUM = zeros(N, K);
    for i = 1:M
        Temp = exp(Soft(:, (i-1)*K+1 : i*K));
        %Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K) / M;
        Temp = Temp ./ repmat(sum(Temp, 2), 1, K) / M;
        Ori  = Temp;
        Temp = exp(X(:, (i-1)*K+1 : i*K));
        Temp = Temp ./ repmat(sum(Temp, 2), 1, K) / M;
        SUM = SUM + Temp + Ori;
    end
    [mx, label] = max(SUM, [], 2);
    fprintf('Strategy 6[exp:sum] : %.5f\n' , mean(label == test_label));
%}

end
