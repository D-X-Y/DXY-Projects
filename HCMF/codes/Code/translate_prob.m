function [prob,x] = translate_prob(M,rank,state)

%prob.mu = 0.01;


if nargin==3
    randn('state', state);
    rand('state', state);
end
prob.r = rank;
prob.k = rank;

prob.n1 = M.size1; 
prob.n2 = M.size2;
prob.m = length(M.ind_Omega);

prob.Omega_i = M.row';
prob.Omega_j = M.col';
prob.data = M.values_Omega(:); % we need a column vector!
prob.Omega = M.ind_Omega';





prob.temp_omega = sparse(prob.Omega_i, prob.Omega_j, ...
    prob.data*1,prob.n1,prob.n2,prob.m);



k = rank;
x.V = randn(prob.n1,k); [x.V,temp] = qr(x.V,0);
x.sigma = sort(abs(randn(k,1)),'descend');
x.U = randn(prob.n2,k); [x.U,temp] = qr(x.U,0);
x = prepx(prob, x);
