function tmin = exact_search_onlyTxM(prob, x,dir)

% Exact line search in the direction of dir on the tangent space of x
% !! so NOT the retracted curve, only use as guess !!

% x, current point

% dir is search direction
%
% returns: tmin

e_omega = x.err;
%dir_omega = XonOmega([x.U*dir.M+dir.Up x.U], [x.V dir.Vp], prob.Omega);

if 2*prob.r >= min(prob.n1, prob.n2)
    dir_omega = partXY(dir.M'*x.U'+dir.Up',x.V', prob.Omega_i, prob.Omega_j, prob.m) + ...
        partXY(x.U',dir.Vp', prob.Omega_i, prob.Omega_j, prob.m); 
else
    dir_omega = partXY([x.U*dir.M+dir.Up x.U]',[x.V dir.Vp]', prob.Omega_i, prob.Omega_j, prob.m); 
end


% norm is f(t) = 0.5*||e+t*d||_F^2
% minimize analytically
% polynomial df/dt = a0+t*a1
a0 = dir_omega*e_omega;
a1 = dir_omega*dir_omega';
tmin = -a0/a1;


% A1 = dir_omega';
% b = e_omega;
% tmin = -A1\b;

