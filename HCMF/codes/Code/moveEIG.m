function z = moveEIG(prob, x,h,t)
%MOVEEIG  Retract a point x+t*h (minimal distance)
%
%  z = moveEIG(sys, x,h,t) retracts x+t*h to the manifold where
%   sys is a system,
%     x is a point of sys,
%     h is a tangent vector of TxM,
%     t is a distance (scalar).
%  Retraction is the shortest distance to the manifold.
%
%  Implemented as a partitioned eigenvalue decomposition with 
%  complexity O(k^3 n).


kc = size(x.V,2);

% qr of Vp and Up only
[U_t,Ru] = qr(t*h.Up,0);
[V_t,Rv] = qr(t*h.Vp,0);


Ru_M_Rv = [diag(x.sigma)+t*h.M Rv'; Ru zeros(kc)];
tk = bestApprox(Ru_M_Rv,kc);
z.U = [x.U U_t]*tk.U;
z.V = [x.V V_t]*tk.V;
z.sigma = tk.sigma;


z = prepx(prob, z);

end

function z = bestApprox(X,k)
%BESTAPPROXRCSPD    Approximate a matrix i
%
%  z = bestApproxRcSpd(X,k) calculates the best approximation 
%  of a matrix X in the set of matrices of rank <= k.
%
%  Returns z with z.U*diag(z.sigma)*z.V' the best approximation.
%
%  Direct method with cubic complexity


[U,S,V] = svd(X);


z.U = U(:,1:k);
z.V = V(:,1:k);
z.sigma = diag(S(1:k,1:k)) + eps;


end