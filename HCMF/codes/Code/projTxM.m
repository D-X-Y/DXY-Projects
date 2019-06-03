function h = projTxM(x,h)
% PROJTXM  Project a tangent vector near TxM to TxM
%
%  h = projTxM(x,h) projects an existing tangent vector h
%  near the tangent space of x to TxM. 
%  Usefull to avoid roundup errors.

h.Vp = h.Vp - x.V * (x.V'*h.Vp);
h.Up = h.Up - x.U * (x.U'*h.Up);