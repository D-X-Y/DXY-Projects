% Example for Call :
% warp_vis(250, 15, 20, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'results/visual/');
% synthetic_bad_good(1000, 10, 20, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
% Figure 2 and Figure 3
% synthetic_bad_good(500, 6, 20, [0.4, 0.4, 0.4, 0.2, 0.2, 0.2]);
% 98.20
% synthetic_bad_good(500, 6, 20, [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]);
function synthetic_bad_good(N, M, Class, error_ratios)


  fprintf('warp the visualization with N=%d, Class=%d, classifiers=%d\n', N, Class, M);
  % visualization 
  pix_rect = [3, 8];
  gt_X = rand(N, Class);


  % generate the synthetic data
  [tX, tLabel, GT, ER] = generate(gt_X, N, M, Class, error_ratios);
  % construct the input matrix
  tL = []; L = [];
  for i = 1:numel(ER)
    L = [L, ER{i}];
    tL = [tL, tX];
  end
  
  % parameters
  p  = 1.01;
  mu = 1;
  max_iters = 2000;
  [~, max_index] = merge_results(L, numel(ER), Class);
  Acc_ours_before = sum(max_index==tLabel) / N * 100;
  [X_ours]  = SolverSFEC(L, numel(ER), Class, p, mu, max_iters);
  [~, max_index] = merge_results(X_ours, numel(ER), Class);
  Acc_ours_after = sum(max_index==tLabel) / N * 100;
    
  fprintf('Accuracy for the error ratio of %.2f : %.2f  ->  %.2f\n', mean(error_ratios), Acc_ours_before, Acc_ours_after);
end

function [fusion, max_index] = merge_results(X, M, Class)
    assert(size(X, 2) == M*Class);
    fusion = zeros(size(X,1), Class);
    for i = 1:M
        fusion = fusion + X(:,(i-1)*Class+1:i*Class);
    end
    fusion = fusion / M;
    [~, max_index] = max(fusion, [], 2);
end