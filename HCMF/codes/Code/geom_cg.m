function [xc,histout,costdata,fail] = geom_cg(prob, opts, x0)
% function [x,histout,costdata] = steep(x0,f,tol,maxit)
%
% Geometric cg descent with Armijo rule, polynomial linesearch
%
%
% Input: x0 = initial iterate
%        f = objective function,
%            the calling sequence for f should be
%            [fout,gout]=f(x) where fout=f(x) is a scalar
%              and gout = grad f(x) is a COLUMN vector
%        tol = termination criterion norm(grad) < tol
%              optional, default = 1.d-6
%        maxit = maximum iterations (optional) default = 1000
%
% Output: x = solution
%         histout = iteration history
%             Each row of histout is
%            [rel_norm(grad), rel_err_on_omega, relative_change, ...
%                number of step length reductions, restarts]
%
% Adapted from steep by C. T. Kelley, Dec 20, 1996
%

%


t_begin = tic();

TOL_RESCHG = 1e-8;

%beta_type = 'F-R';
beta_type = 'P-R';

fail = true;

norm_M_Omega = norm(prob.data);

ORTH_VALUE = 0.1; % the search directions should be almost orthogonal

itc = 1; xc = x0;
fc = F(prob,xc);
gc = grad(prob,xc);
ip_gc = ip(xc,gc,gc);
fold = 2*fc;
% first search-dir is steepest gradient
dir = scaleTxM(gc,-1);

numf = 1; numg = 1; numh = 0;
ithist=zeros(opts.maxit,4);
%ithist(1,1)=sqrt(ip_gc); ithist(1,2) = fc; ithist(1,4)=itc-1; ithist(1,3)=0;
xc_old1 = xc;
beta = 0;


for itc=1:opts.maxit
  tinit(itc) = exact_search_onR1(prob, xc,dir);
  
   
  fc_b = fc;
  [xc_new,fc,succ,numf_a,iarm,tinit(itc)] = armijo_search(prob, xc,fc,dir, tinit(itc));
    
  if ~succ && beta ~= 0
    if opts.verbosity > 0; warning('Line search failed on CG. Resetting to gradient.'); end
    beta = 0;
    % if we did cg, reset to steepest descent and try again
    dir = scaleTxM(gc,-1);
    tinit(itc) = exact_search_onR1(prob, xc,dir);
    %tinit(itc) = exact_search_onlyTxM_multiple(prob, xc, {dir});
    [xc_new,fc,succ,numf_a,iarm,tinit(itc)] = armijo_search(prob, xc,fc_b,dir, tinit(itc));
  end
  
  % if it still fails (beta is always 0 here -> steepest descent)
  % then we give up
  if ~succ
    xc = xc_new;
    ithist(itc,1) = rel_grad;
    ithist(itc,2) = sqrt(2*fc)/norm_M_Omega;
    ithist(itc,3) = reschg;
    ithist(itc,4) = iarm;    
    histout=ithist(1:itc,:); costdata=[numf, numg, numh];
    if opts.verbosity > 0; warning('Line search failed on steepest descent. Exiting...'); end
    return
  end
  
  
  % grad(new x)
  gc_new = grad(prob,xc_new);
  ip_gc_new = ip(xc_new,gc_new,gc_new);
  
  % Test for convergence
  if sqrt(2*fc) < opts.abs_f_tol
    if opts.verbosity > 0
      disp('Abs f tol reached.')
    end
    fail = false;
    break;
  end
  if sqrt(2*fc)/norm_M_Omega < opts.rel_f_tol
    if opts.verbosity > 0; disp('Relative f tol reached.'); end
    fail = false;
    break;
  end
  
  rel_grad = sqrt(ip_gc)/max(1,norm(xc_new.sigma));
  if rel_grad < opts.rel_grad_tol
    if opts.verbosity > 0; disp('Relative gradient tol reached.'); end
    fail = false;
    break;
  end
  
  % for 'stagnation stopping criterion' after 10 iters
  reschg = abs(1-sqrt(2*fc)/sqrt(2*fold) );  % LMARank's detection
  %reschg = abs(sqrt(2*fc) - sqrt(2*fold)) / max(1,norm_M_Omega);
  if itc > 10 && reschg < opts.rel_tol_change_res
    if opts.verbosity > 0; disp('Iteration stagnated rel_tol_change_res.'); end
    fail = true;
    break;
  end
  
  
  if itc > 2
    R1 = triu(qr([xc_new.U*diag(xc_new.sigma) xc.U*diag(xc.sigma)],0));
    R2 = triu(qr([xc_new.V -xc.V],0));
    
    %rel_change_x = norm(xc_new.U*diag(xc_new.sigma)*xc_new.V' - xc.U*diag(xc.sigma)*xc.V', 'fro') / ...
    %  norm(xc_new.U*diag(xc_new.sigma)*xc_new.V', 'fro');
    rel_change_x = norm(R1*R2','fro')/norm(xc_new.sigma,2);

    if rel_change_x < opts.rel_tol_change_x
      if opts.verbosity > 0; disp('Iteration stagnated for rel_tol_change_x.'); end
      fail = true;
      break;
    end
  end
    
  % new dir = -grad(new x)
  %           + beta * vectTransp(old x, old dir, tmin*old dir)
  gc_old = transpVect(prob,xc,gc,xc_new,1);  
  dir = transpVect(prob,xc,dir,xc_new,1);
      
  % we test how orthogonal the previous gradient is with
  % the current gradient, and possibly reset the to the gradient
  orth_grads = ip(xc_new,gc_old,gc_new)/ip_gc_new;
  
  if (orth_grads) >= ORTH_VALUE
    if opts.verbosity > 1; disp('New gradient is almost orthogonal to current gradient. This is good, so we can reset to steepest descent.'); end
    beta = 0;
    dir = plusTxM(gc_new, dir, -1, beta);
    
  else % Compute the CG modification
    % beta
    if strcmp(beta_type, 'F-R')  % Fletcher-Reeves
      beta = ip_gc_new / ip_gc;
      % new dir
      dir = plusTxM(gc_new, dir, -1, beta);
    elseif strcmp(beta_type, 'P-R')  % Polak-Ribiere
      % vector grad(new) - transported grad(current)
      diff = plusTxM(gc_new, gc_old, 1, -1);
      ip_diff = ip(xc_new, gc_new, diff);
      beta = ip_diff / ip_gc;
      
      %dk_c = transpVect(prob, xc,dir,xc_new,1);
      %ct = -1/(sqrt(ip(xc_new,dk_c,dk_c))*min(0.01,sqrt(ip_gc)));
      %beta = max(ct,beta);
      beta = max(0,beta);
      dir = plusTxM(gc_new, dir, -1, beta);
    end
  end
  
  % check if dir is descent, if not take -gradient (i.e. beta=0)
  g0 = ip(xc_new,gc_new,dir);
  if g0>=0
    if opts.verbosity > 1; 
      disp('New search direction not a descent direction. Resetting to -grad.');       
    end
    dir = scaleTxM(gc_new, -1);
    beta = 0;
  end
  
  
  % update _new to current
  gc = gc_new;
  ip_gc = ip_gc_new;
  xc = xc_new;
  fold = fc;
  
  numg = numg+1;
  
  ithist(itc,1) = rel_grad;
  ithist(itc,2) = sqrt(2*fc)/norm_M_Omega;
  ithist(itc,3) = reschg;
  ithist(itc,4) = iarm;    
end

xc = xc_new;
ithist(itc,1) = rel_grad;
ithist(itc,2) = sqrt(2*fc)/norm_M_Omega;
ithist(itc,3) = reschg;
ithist(itc,4) = iarm; 
histout=ithist(1:itc,:); 
