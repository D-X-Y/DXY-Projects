function Y = Ncut2Res(X, M, K)
    assert(false); % Do not use Ncut
    [XX, ~] = GetTrueLabel(X, M);
    assert(size(XX,1) == size(X,1));
    assert(size(XX,2) == K);
    [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(X * X', K);
    [~, NcutClusters] = max(NcutDiscrete, [], 2);
    used = zeros(K, 1);
    mapping = zeros(K, 1);
    Y = zeros(size(X,1), 1);
    for index = 1:K
        idxs = find(NcutClusters == index);
        T_Temp = XX(idxs, :);
        [~, T_Label] = sum(T_Temp);
        [values, ords] = sort(SUM, 'descend');
        for jj = 1:K
            true_label = ords(jj);
            if (used(true_label) == 0) break; end
        end
        assert(used(true_label) == 0);
        used(true_label) = 1;
        Y(idxs) = true_label;
    end
end
function [ords, values] = GetHightValue(T_Temp)
    SUM = sum(T_Temp);
    %[values, ords] = sort(SUM, 'descend');


end
