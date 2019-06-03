function [opts] = default_alm()
    opts.K = -1;
    % LRGeomCG : OS
    opts.p = 1.1;
    opts.mu = 0.1;
    opts.max_iters = 100;
end
