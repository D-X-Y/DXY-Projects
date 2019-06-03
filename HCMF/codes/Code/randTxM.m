function h = randTxM(x)
%RANDTXM   Generate a random tangent vector
%
% h = randTxM(x) generates a random tangent vector in the tangent 
% space of x.


[n1,k] = size(x.U);
[n2,k] = size(x.V);

h.M  = rand(k,k);
h.Vp = rand(n2,k); 
h.Up = rand(n1,k);
h = projTxM(x,h);