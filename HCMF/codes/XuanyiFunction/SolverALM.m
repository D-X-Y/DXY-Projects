function [X, param] = SolverALM(results, label, opts)
K = opts.K;
assert( K >= 0);
% results is N * M matrix, donate M classifier's results
time_begin = tic;
[N, M] = size(results);
assert( N == size(label, 1));
assert( N == numel(label));
assert( 2 == numel(size(results)));
fprintf('Assert OK, N : %d, K : %d, M : %d( check dimisions of matrix)\n', N, K, M);
% First Convert results to cluster assignments
L = zeros(N, 0);
for index = 1:M
    ID = results(:, index);
    L_temp = zeros(N, K)';
    Gap = (0:N-1)*K;
    L_temp(ID+Gap') = 1;
    L = [L, L_temp'];
    % For Check 
    [mxx, idd] = max(L_temp', [], 2);
    assert(all(idd == ID));
end
fprintf('Initialize L done\n');
% Apply ALM Optimization 
% min ||X-L||_(1,2) , s.t. rank(X) <= k, s.t X = UV, U \belong R(N*k)
% The augmented Lagrangian function:
% Lagrangian = ||E||1,2 + <lamda, L-X-E> + mu/2*||L-X-E||^2_F
% lamda is a MK * N matrix

% LRGeomCG : OS
% Initialize 
lamda = zeros(N, M*K); E = zeros(N, M*K);
%max_iters = 100;
max_iters = opts.max_iters;
for iter = 1:max_iters
    fprintf('%04d / %04d iterations start.....\n', iter, max_iters);
    tic;
    % Obtain X(t) via LRGeomCG
    X_verse = L - E + (lamda / opts.mu);
    X = wrapCG(X_verse, K, opts.options);
    % Obtain E(t) via soft-thresholding
    Y = L - X + (lamda / opts.mu);
    alpha = 2 / opts.mu;
    E = zeros(N, M*K);
    parfor col = 1:M*K
        norm2 = norm(Y(:,col));
        if norm2 > alpha
            E(:,col) = Y(:,col) * (1-alpha/norm2);
        end
    end
    lamda = lamda + opts.mu * (L - X - E);
    opts.mu = opts.mu * opts.p;
    fprintf ('norm(L) : %.4f, norm(X) : %.4f, norm(L-X) : %.4f, alpha : %.4f\n', norm(L, 'fro'), norm(X, 'fro'), norm(L-X,'fro'), alpha);
    fprintf ('norm(E) : %.4f, norm(lamda) : %.4f, p : %.2f, mu %.4f\n', norm(E, 'fro'), norm(lamda, 'fro'), opts.p, opts.mu);
    [SCORES, Tlabel] = GetTrueLabel(X, M, K);
    fprintf('Current Accuracy : %.5f\n', sum(Tlabel == label) / numel(label));
    fprintf ('%04d / %04d iterations done , cost %.3f s ***\n\n', iter, max_iters, toc);
    diary; diary; % flush diary
end
param.L = L;
param.E = E;
param.X = X;
param.lamda = lamda;
param.K  = K;
param.M  = M;
param.opts = opts;
fprintf('Total iteration : %d, Cost : %.3f s\n', max_iters, toc(time_begin));

end
