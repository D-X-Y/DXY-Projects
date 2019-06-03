% Example for Call :
% warp_vis(250, 15, 20, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'results/visual/');
% warp_vis(250, 15, 20, [0.3], 'results/visual/');
% Figure 2 and Figure 3
function warp_vis(N, M, Class, error_ratios, save_dir)
  fprintf('save the results into %s', save_dir);
  status = mkdir(save_dir);
  assert (status, 'The mkdir state is wrong');
  fprintf('warp the visualization with N=%d, Class=%d, classifiers=%d\n', N, Class, M);
  % visualization 
  pix_rect = [3, 8];
  gt_X = rand(N, Class);


  for ie = 1:numel(error_ratios)
    error_ratio = error_ratios(ie);
    indicator = num2str(error_ratio);
    % generate the synthetic data
    [tX, tLabel, GT, ER] = generate(gt_X, N, M, Class, error_ratio);
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
    % compare with RCEC ::: beta = {0.01, 0.1, 1, 2, 5, 10}
    opts_rcec = struct('K', Class, 'max_iters', max_iters, 'beta', 1, 'gamma', 0.01, 'lamda', 0.1);
    [X_rcec]  = Solver_Rcec(L, tLabel, opts_rcec);

    % pca 
    %[_, _, latent_] = pca(tL);
    all_cells = cell(4, 3);
    all_cells{1, 1} = tL;     all_cells{1, 2} = 'Ground Truth';                         all_cells{1, 3} = 'g-o';
    all_cells{2, 1} = X_rcec; all_cells{2, 2} = 'RCEC';                                 all_cells{2, 3} = 'r-o'; 
    all_cells{3, 1} = X_ours; all_cells{3, 2} = 'Ours';                                 all_cells{3, 3} = 'g-o';
    all_cells{4, 1} = L;      all_cells{4, 2} = 'Results of Synthetic Classifiers';     all_cells{4, 3} = 'b-o';
    all_cells = all_cells(2:4, :);
    save_pca(all_cells, strcat(save_dir, '/', 'curve-', indicator, '.pdf'))

    % visualization
    image_ours  = obtain_image(X_ours, M, Class, pix_rect);
    image_rcec  = obtain_image(X_rcec, M, Class, pix_rect);
    image_error = Convert2Image(ER, pix_rect);
    image_gt    = Convert2Image(GT, pix_rect);

    save_image(image_ours , strcat(save_dir, '/', 'ours-', indicator, '.pdf'));
    save_image(image_rcec , strcat(save_dir, '/', 'rcec-', indicator, '.pdf'));
    save_image(image_error, strcat(save_dir, '/', 'eror-', indicator, '.pdf'));
    save_image(image_gt   , strcat(save_dir, '/', 'grot-', indicator, '.pdf'));
    fprintf('Accuracy for the error ratio of %.2f : %.2f  ->  %.2f\n', error_ratio, Acc_ours_before, Acc_ours_after);
  end

end

function [fusion, max_index] = merge_results(X, M, Class)
    assert(size(X, 2) == M*Class);
    fusion = zeros(size(X,1), Class);
    for i = 1:M
        fusion = fusion + X(:,(i-1)*Class+1:i*Class);
    end
    fusion = fusion / M;
    [Max, max_index] = max(fusion, [], 2);
end

function cells = split(Matrix, M, Class)
  cells = cell(M, 1);
  for i = 1:M
    cells{i} = Matrix(:, (i-1)*Class+1 : i*Class);
  end
end

function image = obtain_image(X, M, Class, pix_len)
  size_t = size(X);
  assert (size_t(2) == M * Class);
  reconstruction = split(X, M, Class);
  image = Convert2Image(reconstruction, pix_len);
end

function save_image(image, path)
  imagesc(image);
  set(gca,'XTick',[]); % Remove the ticks in the x axis!
  set(gca,'YTick',[]); % Remove the ticks in the y axis
  %set(gca,'Position',[0 0 1 1]); % Make the axes occupy the hole figure
  print(gcf, path,'-dpdf','-r300');
  %system(['pdfcrop ', path, ' ', path]);
  close all;
end

function [ANS] = for_debug(ER, GT, matrix_A, matrix_B, Classes, i_cls, j_index)
  size_t = size(matrix_A, 2);
  assert ( numel(ER) == numel(GT) );
  assert ( size_t == Classes * numel(ER), strcat(num2str(size_t), ', ', num2str(numel(ER))) );
  num_classifier = numel(ER);
  er = ER{i_cls};   gt = GT{i_cls};
  AA = matrix_A(:, (i_cls-1)*Classes+1 : i_cls*Classes);
  BB = matrix_B(:, (i_cls-1)*Classes+1 : i_cls*Classes);
  ANS = [er(j_index,:); AA(j_index,:); BB(j_index,:); gt(j_index,:)];
end

function save_pca(cells, path)
  num = size(cells, 1);
  close all;
  threshold = 50;
  for i = 1:num
    [COEFF,SCORE,latent] = pca( cells{i, 1} );
    %latent = cumsum(latent)./sum(latent);
    latent = latent ./ sum(latent);
    plot( (1:threshold), latent(1:threshold), cells{i, 3}, 'LineWidth', 4 );
    hold on;
  end
  font_size = uint8(20);
  legend(cells(:, 2) , 'FontSize', font_size);
  %ylabel('the ratio of the largest eigenvalues to the sum of eigenvalues', 'FontSize', font_size);
  %xlabel('the index of each eigenvalue (sorted from large to small)', 'FontSize', font_size);
  ylim([0, 0.1]);
  set(gca,'fontsize', font_size);
  print(gcf, path, '-dpdf','-r300');
  %system(['pdfcrop ', path, ' ', path]);
  fprintf('save into %s\n', path);
  close all;
end
