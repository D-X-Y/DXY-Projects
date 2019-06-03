function Eta = vecTxM(x,eta)
%VECTXM  Vectorize a tangent vector to normal array
%
%   Store eta as a normal array (column ordering)
%     Eta = [eta.S eta.Yp']



Eta = [eta.M eta.Up' eta.Vp'];
Eta = Eta(:);