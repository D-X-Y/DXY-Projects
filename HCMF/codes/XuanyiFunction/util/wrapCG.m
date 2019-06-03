function Xcg_est = wrapCG(X, k, options)
[m, n] = size(X);
%k = 40; % rank
% Relative oversampling factor 
% OS=1 is minimum, 2 is difficult, 3 is OKish, 4+ is easy.
% OS = 4;

% random factors
%L = randn(m, k); 
%R = randn(n, k); 
%X = L * R';
%dof = k*(m+n-k);

tic;
% make random sampling, problem and initial guess
%samples = floor(OS * dof);
%Omega = make_rand_Omega(m,n,samples);
Omega = (1:m*n)';
% prob = make_prob(L,R,Omega,k); % <- you can choose another rank here
prob = get_prob(X,Omega,k);
sample_cost = toc;

% options
% options = default_opts();
% options.maxit = 200;
% Tolerance on the Riemannian gradient of the objective function
% options.abs_grad_tol = 1e-3;
% options.rel_grad_tol = 1e-3;
% Tolerance on the l_2 error on the sampling set Omega
% options.abs_f_tol = 1e-3;
% options.rel_f_tol = 1e-3;

x0 = make_start_x(prob);

t=tic;
[Xcg,hist] = LRGeomCG(prob,options,x0);

Xcg_est = Xcg.U * diag(Xcg.sigma) * Xcg.V';
error_cg = norm(Xcg_est-X, 'fro') / norm(X, 'fro');
fprintf('LRGeomCG : error_cg : %.12f, cost [%.3f , %.3f] s\n', error_cg, sample_cost ,toc(t));
end
