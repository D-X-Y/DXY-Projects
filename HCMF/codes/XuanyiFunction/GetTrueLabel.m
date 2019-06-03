function [SUM, label] = GetTrueLabel(X, M, K)
    assert(size(X, 2) == M*K);
    N = size(X,1);
    SUM = zeros(N, K);
    for i = 1:M
        Temp = X(:, (i-1)*K+1 : i*K);
        Temp = Temp ./ repmat(sum(Temp.*Temp, 2), 1, K);
        SUM = SUM + Temp;
    end
    SUM = SUM / M;
    [mx, label] = max(SUM, [], 2);
end
