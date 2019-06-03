function [xt,ft, succ,numf,iarm, t] = exact_search(sys, xc,fc,gc,tl,tr)

% Exact ('numerically exact') line search on an interval with fminbnd.

% xc, current point
% fc = F(sys,xc)
% gc is search direction
%
% returns: xt,ft: local minimum point and F(sys,xt)
%          succ = true if success
%          numf: nb of F evals
%          iarm: nb of Armijo backtrackings

opts.MaxIter = 100;
opts.TolX = 1e-10;



%tr = tr*1e5;

%ts = linspace(0,tr,500);
%fs = fun(sys,ts,xc,gc);
%semilogy(ts,fs,'.')
%pause
    function ft = fun(lambda)
        xt = moveEIG(sys,xc,gc,lambda);
        ft = F(sys,xt);
    end

[t,ft,flag,output] = fminbnd(@fun,tl,tr,opts);
xt = moveEIG(sys,xc,gc,t);
if flag == 1
    succ = true;
else
    succ = false;
end
numf = output.funcCount;
iarm = output.iterations;

end