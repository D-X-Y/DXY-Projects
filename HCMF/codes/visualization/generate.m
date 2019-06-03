function [X, aY, GT, ER] = generate(X, N, M, K, error_ratios)
  %X = rand(N, K);
  if numel(error_ratios) == 1
      error_ratios = zeros(M, 1) + error_ratios;
  end
  assert(numel(error_ratios) == M);
  [tY, aY] = max(X, [], 2);
  X = X == repmat(tY, 1, K);
  FinalX = cell(M, 1);
  for index = 1:M
    error_ratio = error_ratios(index);
    idx = randperm(N, ceil(error_ratio*N));
    Y = X;
    for i = 1:numel(idx)
        Y(idx(i),:) = 0;
        Y(idx(i),randperm(K,1)) = 1;
    end
    FinalX{index} = Y;
  end
  ER = FinalX;
  GT = cell(M, 1);
  for i = 1:M, GT{i} = X; end
end
