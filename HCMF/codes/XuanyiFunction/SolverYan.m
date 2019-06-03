function [X, param] = SolverYan(results, label, opts)
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

% Initialize LRGeomCG
Omega = (1:N*M*K)';
prob = get_prob(L,Omega,K);
x0 = make_start_x(prob);
previous_norm = -100;
epsilon = 1e-4;

% Initialize 
lamda = zeros(N, M*K); E = zeros(N, M*K);
%max_iters = 100;
max_iters = opts.max_iters;

for iter = 1:max_iters
    fprintf('%04d / %04d iterations start.....\n', iter, max_iters);
    tic;
    % Obtain X(t) via LRGeomCG
    X_verse = L - E + (lamda / opts.mu);
    % X = wrapCG(X_verse, K, opts.options);
    prob.data = reshape(X_verse, N*M*K, 1);
    prob.temp_omega = sparse(prob.Omega_i, prob.Omega_j, prob.data*1,prob.n1,prob.n2,prob.m);
    [x0, ~] = LRGeomCG(prob,opts.options,x0);
    X = x0.U * diag(x0.sigma) * x0.V';

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

    temp = norm(E, 'fro');
    termination = false;
    change_ratio = abs(temp-previous_norm) / abs(previous_norm);
    if (change_ratio < epsilon);
        termination = true;
    else
        previous_norm = temp;
    end
    fprintf ('PreNorm: %.6f, CurNorm : %.6f\n abs_res : %.7f , done [%.3f] s\n\n',previous_norm, temp, change_ratio, toc);
    if (termination)
        fprintf('End Break in %d\n', iter); break;
    end
    diary; diary; % flush diary
end
time_cost = toc(time_begin);
fprintf('Total iteration : %d, Cost : %.3f s\n', max_iters, time_cost);
param.L = L;
param.E = E;
param.X = X;
param.lamda = lamda;
param.K  = K;
param.M  = M;
param.opts = opts;
param.time_cost = time_cost;

end
