function [X] = SolverSFEC(L, M, K, p, mu, max_iters)
% results is N * M matrix, donate M classifier's results
time_begin = tic;
assert( size(L,2) == M * K );
% First Convert results to cluster assignments
% Apply ALM Optimization 
% min ||X-L||_(1,2) , s.t. rank(X) <= k, s.t X = UV, U \belong R(N*k)
% The augmented Lagrangian function:
% Lagrangian = ||E||1,2 + <lamda, L-X-E> + mu/2*||L-X-E||^2_F
% lamda is a MK * N matrix
N = size(L, 1);
M = size(L, 2);
% LRGeomCG :::: Initialize 
lamda = zeros(N, M); E = zeros(N, M);
%max_iters = 100;
fprintf('Initialize paramters: p : %.4f, mu : %.4f, size(L) : [%d,%d]\n',p,mu,size(L));

% Initialize LRGeomCG
Omega = (1:N*M)';
prob = get_prob(L,Omega,K);
x0 = make_start_x(prob);
options = default_opts();

previous_norm = -100;
time_begin = tic;
for iter = 1:max_iters
    tic;
    fprintf('%03d/%03d :: ', iter, max_iters);
    % Obtain X(t) via LRGeomCG
    X_verse = L - E + (lamda / mu);
    %X = wrapCG(X_verse, K);
    %prob = get_prob(X_verse,Omega,K);
    prob.data = reshape(X_verse, numel(L), 1);
    prob.temp_omega = sparse(prob.Omega_i, prob.Omega_j, prob.data*1,prob.n1,prob.n2,prob.m);
    %data_time = toc;
    %assert(isempty(find(AA~=BB)));
    [x0,~] = LRGeomCG(prob, options,x0);
    %CG_time = toc;
    X = x0.U * diag(x0.sigma) * x0.V';
    
    % Obtain E(t) via soft-thresholding
    Y = L - X + (lamda / mu);
    alpha = 2 / mu;
    E = zeros(N, M);
    for col = 1:M
        norm2 = norm(Y(:,col));
        if norm2 > alpha
            E(:,col) = Y(:,col) * (1-alpha/norm2);
        end
    end
    lamda = lamda + mu * (L - X - E);
    mu = mu * p;
    temp = norm(E, 'fro');
    termination = false;
    if (abs(temp-previous_norm) / abs(previous_norm) < 1e-5)
        termination = true;
    else
        previous_norm = temp;
    end
    fprintf ('PreNorm: %.3f, CurNorm : %.3f, done [%.3f] s\n',previous_norm, temp, toc);
    if (termination)
        fprintf ('End Break in %d\n', iter); break;
    end
    %fprintf ('norm(L) : %.4f, norm(X) : %.4f, norm(E) : %.4f \n\n', norm(L, 'fro'), norm(X, 'fro'), norm(E, 'fro'));
end

fprintf('Total iteration : %d, Cost : %.3f s\n', max_iters, toc(time_begin));

end
