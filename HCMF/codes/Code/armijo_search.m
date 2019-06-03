function [xt,ft, succ,numf,iarm,lambda] = armijo_search(prob, xc,fc,gc,dir,lambda)

% Armijo line search with polynomial interpolation
% Adapted from steep by C. T. Kelley, Dec 20, 1996

% xc, current point
% fc = F(sys,xc)
% gc is gradient
% dir is search direction
%
% returns: xt,ft: Armijo point and F(sys,xt)
%          succ = true if success
%          numf: nb of F evals
%          iarm: nb of Armijo backtrackings




%%% constants
alp=1e-4; blow=0.1; bhigh=0.5;
%alp = 1e-8; bhigh=.5; blow=.1;
%%%
MAX_IARM = 3;

numf = 0;

% trial evaluation at full step
xt = moveEIG(prob,xc,dir,lambda);
ft = F(prob,xt);
numf = numf+1; iarm = 0;

%fgoal = fc-alp*lambda*ip(xc,gc,gc); %norm(gc.Y,2)^2;
%
%       polynomial line search
%
q0=fc; qp0=-ip(xc,gc,dir); lamc=lambda; qc=ft;
fgoal=fc+alp*lambda*qp0;

while(ft > fgoal)    
    iarm=iarm+1;
    if iarm==1
        lambda=polymod(q0, qp0, lamc, qc, blow, bhigh);
    else
        lambda=polymod(q0, qp0, lamc, qc, blow, bhigh, lamm, qm);
    end
    qm=qc; lamm=lamc; lamc=lambda;
    xt = moveEIG(prob,xc,dir,lambda);
    ft = F(prob,xt);
    numf = numf+1; qc=ft;
    if(iarm > MAX_IARM)
        succ = false;        
        return
    end    
    fgoal=fc+alp*lambda*qp0;
end
succ = true;


end
