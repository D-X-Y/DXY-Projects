function opts = default_opts()

opts.maxit = 100;

% Tolerance on the Riemannian gradient of the objective function
opts.abs_grad_tol = 0;
opts.rel_grad_tol = 1e-12;

% Tolerance on the l_2 error on the sampling set Omega
opts.abs_f_tol = 0;
opts.rel_f_tol = 1e-12;

% Tolerance for detection of stagnation. 
opts.stagnation_detection = false;  % <--- cheap test but does take some time
opts.rel_tol_change_x = 1e-12;
opts.rel_tol_change_res = 1e-6;

% Verbosity 2 is very chatty.
opts.verbosity = 1;

