function h = scaleTxM(h,a)
% SCALETXM  scale a tangent vector h by a
%

%%
h.M  = a*h.M;
h.Up = a*h.Up;
h.Vp = a*h.Vp;