function [X, param] = SolverRCEC(results, label, opts)
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
max_iters = opts.max_iters;
X = rand(size(L));
gamma = opts.gamma;
lamda = opts.lamda;
beta  = opts.beta;
dims = size(L, 2);

previous_norm = -100;
epsilon = 1e-5;
for iter = 1:max_iters
    fprintf('%04d / %04d iterations start.....\n', iter, max_iters);
    tic;
    D = diag(1 ./ sqrt(sum(X.*X, 1)));
    H = (X' * X + gamma * eye(dims))^(-0.5);
    H_plus = (abs(H)+H) / 2;
    H_minu = (abs(H)-H) / 2;
    A = 2 * L + lamda * X * H_minu;
    B = 2 * X + beta * X * D + lamda * X * H_plus;
    X = X .* sqrt( A ./ B );

    termination = false;
    temp = norm(L-X, 'fro');
    if (abs(temp-previous_norm) / abs(previous_norm) < epsilon);
        termination = true;
    else
        previous_norm = temp;
    end
    fprintf ('PreNorm: %.5f, CurNorm : %.5f, done [%.3f] s\n',previous_norm, temp, toc);
    if (termination)
        fprintf('End Break in %d\n', iter); break;
    end

    %fprintf ('norm(L) : %.4f, norm(X) : %.4f, norm(L-X) : %.4f\n', norm(L, 'fro'), norm(X, 'fro'), norm(L-X,'fro'));
    %[SCORES, Tlabel] = GetTrueLabel(X, M, K);
    %fprintf('gamma : %.2f, lamda : %.2f, beta : %.2f == Current Accuracy : %.5f\n', gamma, lamda, beta, mean(Tlabel == label));
    %fprintf ('%04d / %04d iterations done , cost %.3f s ***\n\n', iter, max_iters, toc);
    diary; diary; % flush diary
end
time_cost = toc(time_begin);
fprintf('Total iteration : %d, Cost : %.3f s\n', max_iters, time_cost);
param.L = L;
param.X = X;
param.opts = opts;
param.M = M;
param.K = K;
param.E = X-L;
param.time_cost = time_cost;

end
