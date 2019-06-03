function x = prepx(prob, x)
%PREPX  Prepare a point
%
%  Prepare a point x on the manifold to include extra information by
%  computing
%   

if isempty(x.sigma)
    x.on_omega = zeros(size(prob.Omega));
else
    x.on_omega = partXY(diag(x.sigma)*x.U',x.V', prob.Omega_i, prob.Omega_j, prob.m)'; 
end


x.err = x.on_omega - prob.data;
