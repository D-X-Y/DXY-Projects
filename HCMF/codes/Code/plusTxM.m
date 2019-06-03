function h = plusTxM(h1,h2,a1,a2)
% PLUSTXM  add 2 tangent vectors in the same tangent space
%
%  h = a1*h1 + a2*h2  a1,a2 reals

%%
h.M = a1*h1.M + a2*h2.M;
h.Up = a1*h1.Up + a2*h2.Up;
h.Vp = a1*h1.Vp + a2*h2.Vp;