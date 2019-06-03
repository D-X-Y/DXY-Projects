% [boxes, images] = generate_noise_proposals('./data/noise-list.mat');
function [boxes, images] = generate_noise_proposals(mat_path)
  mat = load(mat_path);
  file_list = mat.listing;
  file_dirs = mat.dir_path;
  num_file = numel(file_list);
  fprintf('The total noise images : %d\n', num_file);
  fast_mode = true;
  boxes = cell(1, num_file);
  images = cell(num_file, 1);
  print_gap = 100;
  begin_time = tic;
  for i = 1:num_file
    image_start_time = tic;
    image = fullfile(pwd, file_dirs, file_list(i).name);
    assert(exist(image, 'file') == 2);
    image = imread(image);
    boxes{i} = selective_search_boxes(image, fast_mode);
    images{i} = file_list(i).name(1:end-4);
    if rem(i, print_gap) == print_gap-1
      fprintf('%3d / %3d complete, image size : [%3d,%3d], cost %.1f \n', i, num_file, size(image,1), size(image,2), toc(image_start_time));
    end
  end
  end_time = toc(begin_time);
  fprintf('%d images extract SS proposals done in %.1f\n', num_file, end_time);
  save(fullfile('data', 'selective_search_data', 'noise-yfcc100m-1w.mat'), 'boxes', 'images', '-v7.3');
end
