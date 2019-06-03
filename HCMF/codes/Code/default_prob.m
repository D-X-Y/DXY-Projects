function [prob,x] = default_prob()

randn('state',2009);
rand('state',2009);

prob.n1 = 20; prob.n2 = 20; prob.r = 2;
oversampling = 2; 

prob.tau = 0;

prob.M_exact_A = randn(prob.n1,prob.r);
prob.M_exact_B = randn(prob.n2,prob.r);

df = prob.r*(prob.n1+prob.n2-prob.r);
prob.m = min(oversampling*df,round(.99*prob.n1*prob.n2) );

prob.Omega = randsample(prob.n1*prob.n2,prob.m);
prob.Omega = sort(prob.Omega);
%[temp,prob.Omega_indx] = sort(prob.Omega);
[prob.Omega_i, prob.Omega_j] = ind2sub([prob.n1,prob.n2], ...
    prob.Omega);


prob.data =  XonOmega(prob.M_exact_A, prob.M_exact_B, prob.Omega);
prob.M_Omega = sparse(prob.Omega_i, prob.Omega_j, ...
    prob.data,prob.n1,prob.n2,prob.m);
%full(prob.M_Omega(1:5,1:5))
%updateSparse(prob.M_Omega, prob.data, prob.Omega_indx);
%full(prob.M_Omega(1:5,1:5))
prob.temp_omega = sparse(prob.Omega_i, prob.Omega_j, ...
    prob.data*1,prob.n1,prob.n2,prob.m);

m_df = prob.m / df
m_n_n = prob.m / prob.n1 / prob.n2

% check 
%T = zeros(n1,n2);
%M = prob.M_exact_A*prob.M_exact_B';
%T(prob.Omega) = M(prob.Omega);
%norm(prob.M_Omega - T,'fro')

x = make_rand_x(prob);

