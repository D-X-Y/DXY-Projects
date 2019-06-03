function eta = unvecTxM(x,Eta)
%UNVECTXM  Unvectorize a normal array to a tangent vector
%
% Restore eta from a normal array (column ordering)
%  Eta = [eta.S eta.Yp']
%  


k = length(x.sigma);
n = size(x.V,1);

Eta = reshape(Eta, k, n+n+k);
eta.M  = Eta(1:k,1:k);
eta.Up = Eta(1:k,k+1:n+k)';
eta.Vp = Eta(1:k,k+n+1:n+n+k)';

eta.Up = eta.Up - x.U*(x.U'*eta.Up);
eta.Vp = eta.Vp - x.V*(x.V'*eta.Vp);
        
