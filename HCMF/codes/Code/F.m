function f = F(prob,x)
%F Cost function 
%

f = 0.5*(x.err'*x.err);

if prob.mu>0
  f = f + prob.mu*norm(1./x.sigma)^2;
end