function g = grad(prob,x)
%GRAD   Computes the gradient on the manifold
%
%  Computes the gradient at a point x of the cost function on 
%  the manifold.


do_updateSval(prob, prob.temp_omega, x.err, length(x.err));
prob.temp_omega = prob.temp_omega;

T = prob.temp_omega*x.V;
g.M = x.U'*T; 
g.Up = T - x.U*(x.U'*T);
g.Vp = prob.temp_omega'*x.U; 
g.Vp = g.Vp - x.V*(x.V'*g.Vp);

    
    

