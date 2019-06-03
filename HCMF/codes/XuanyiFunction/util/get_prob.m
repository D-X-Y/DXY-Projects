function prob = get_prob(X,Omega,k)
% L : m * k , R : n * k
% L*R' is the exact data
% Omega is the sampling set; a sparse matrix.
% k is the rank to reconstruct

prob.use_blas = true;

prob.n1 = size(X,1); 
prob.n2 = size(X,2); 
prob.r = k;

if prob.r > min(prob.n1, prob.n2)
    error('Rank should be an integer between 1 and the number of columns/rows.')
end

prob.Omega = Omega;
[prob.Omega_i, prob.Omega_j] = ind2sub([prob.n1,prob.n2], Omega);
prob.m = length(Omega);

%prob.data = partXY(L', R',prob.Omega_i, prob.Omega_j,prob.m)'; 
prob.data = zeros(prob.m, 1);
for index = 1:prob.m
    prob.data(index) = X(prob.Omega_i(index), prob.Omega_j(index));
end

prob.temp_omega = sparse(prob.Omega_i, prob.Omega_j, ...
    prob.data*1,prob.n1,prob.n2,prob.m);
  
% regularization parameter
%prob.mu = 1e-14;
prob.mu = 0;



