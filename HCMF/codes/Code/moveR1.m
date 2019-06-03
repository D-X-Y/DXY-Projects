function z = moveR1(prob, x,h,t,extra)
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


if nargin == 4
    extra = 0;
end

kc = size(x.V,2);

S = diag(x.sigma);
Sinv = diag(1./x.sigma);
Zu = x.U*(S+t*0.5*h.M) + t*h.Up;
Zv = x.V + t*0.5*x.V*h.M'*Sinv + t*h.Vp*Sinv;


[Qu,Ru] = qr(Zu,0);
[Qv,Rv] = qr(Zv,0);
[U,S,V] = svd(Ru*Rv');
z.U = Qu*U;
z.V = Qv*V;
z.sigma = diag(S);


z = prepx(prob, z);

