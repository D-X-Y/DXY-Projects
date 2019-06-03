function image_roidb_train = generate_pure(conf, imdb_train, roidb_train, box_param, use_flipped, path_col)

    [image_roidb_train] = weakly_prepare_image_roidb(conf, imdb_train, roidb_train, box_param.bbox_means, box_param.bbox_stds);

    filtered_image_roidb_train = [];

    for index = 1:numel(image_roidb_train)
        gt = image_roidb_train(index).GT_Index;
        is_flip = false;
        if (use_flipped == false)
            flip = -1;
            image_id = image_roidb_train(index).image_id;
        elseif (strcmp(image_roidb_train(index).image_id(end-3:end), 'flip') == 1)
            flip = index-1;
            is_flip = true;
            assert (strcmp([image_roidb_train(flip).image_id, '_flip'], image_roidb_train(index).image_id) == 1);
	          image_id = image_roidb_train(index).image_id(1:end-5);
        else
            flip = index+1;
            try
              assert (strcmp([image_roidb_train(index).image_id, '_flip'], image_roidb_train(flip).image_id) == 1);
            catch
              fprintf('---%d', index);
            end
      	    image_id = image_roidb_train(index).image_id;
        end

        hdf5_path = fullfile(path_col, [image_id, '.h5']);
        corloc    = get_hdf5(hdf5_path);

        Struct = struct('image_path', image_roidb_train(index).image_path, ...
                        'image_id',   image_roidb_train(index).image_id, ...
                        'imdb_name',  image_roidb_train(index).imdb_name, ...
                        'im_size',    image_roidb_train(index).im_size, ...
                        'overlap',    [], ...
                        'boxes',      image_roidb_train(index).boxes(~gt, :), ...
                        'bbox_targets', [], ...
                        'pseudo_boxes', [], ...
                        'Debug_GT_Cls', image_roidb_train(index).class(gt, :), ...
                        'Debug_GT_Box', image_roidb_train(index).boxes(gt, :), ...
                        'image_label',  image_roidb_train(index).image_label, ...
                        'flipid',       flip, ...
                        'is_flip',      is_flip, ...
                        'index',        index, ...
                        'corloc',       convert_corloc(image_roidb_train(index).im_size, corloc, is_flip));
        filtered_image_roidb_train{end+1} = Struct;
    end
    image_roidb_train = cat(1, filtered_image_roidb_train{:});
    check_corloc(conf, image_roidb_train, 0.5);
end

function corloc = get_hdf5(files)
   score = h5read(files, '/score');
   rois  = h5read(files, '/rois')';
   labels= find(h5read(files, '/labels')'== 1);
   gt_boxes = h5read(files, '/gt_boxes')';
   corloc = struct('score', score, 'rois', rois, 'labels', labels, 'gt_boxes', gt_boxes);
end

function corloc_info = convert_corloc(im_size, corloc_info, is_flip)
   rois = corloc_info.rois;
   gt_boxes = corloc_info.gt_boxes;
   if (is_flip)
     gt_boxes(:, [2,4]) = im_size(2) + 1 - gt_boxes(:, [4,2]);
     rois(:, [1,3])     = im_size(2) + 1 - rois(:, [3,1]);
   end
   assert (size(gt_boxes, 2) == 5); 
   assert (size(rois, 2) == 4); 

   corloc_info.rois = rois;
   corloc_info.gt_boxes = gt_boxes;
end


function check_corloc(conf, image_roidb_train, corlocThreshold)
  num_image = numel(image_roidb_train);
  num_class = numel(conf.classes);  
  res = zeros(num_class, 1); 
  for cls = 1:num_class
    overlaps = [];
    for idx = 1:num_image
      gt_boxes = image_roidb_train(idx).Debug_GT_Box( image_roidb_train(idx).Debug_GT_Cls == cls, :);
      %gt_boxes = image_roidb_train(idx).corloc.gt_boxes;
      %gt_boxes = gt_boxes(gt_boxes(:,1) == cls, 2:5);
      if (isempty(gt_boxes)), continue; end

      score = image_roidb_train(idx).corloc.score(:, cls);
      assert (size(image_roidb_train(idx).corloc.score, 2) == num_class);
      [~, mx] = max(score);
      localizedBox = image_roidb_train(idx).corloc.rois(mx, :);
      %assert (size(score, 1) == size(localizedBox, 1));

      overlap = iou(gt_boxes, localizedBox);
      overlap = max(overlap);
      if (overlap >= corlocThreshold),  overlaps(end+1) = 1;
      else,                             overlaps(end+1) = 0; end
    end
    res(cls) = mean(overlaps);
  end
  fprintf('Results:\n');
  res = res * 100;
  for idx = 1:numel(res)
    fprintf('%12s : corloc : %5.2f\n', conf.classes{idx}, res(idx));
  end
  fprintf('\nmean corloc : %.4f\n', mean(res));
end
