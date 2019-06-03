function new_image_roidb_train = get_init_filter(conf, image_roidb_train, box_param)
    num_image = numel(image_roidb_train);
    num_class = numel(conf.classes);
    tic;
    
    a_image_roidb_train = [];
    for i = 1:num_image
      pseudo_boxes = check_filter_img(image_roidb_train(i).pseudo_boxes, image_roidb_train(i).image_label);
      if (isempty(pseudo_boxes)), continue; end
      a_image_roidb_train{end+1} = image_roidb_train(i);
      a_image_roidb_train{end}.pseudo_boxes = pseudo_boxes;
    end

    a_image_roidb_train = cat(1, a_image_roidb_train{:});

    lower_score = [1.0, 1.0, 1.0, 1.0, 1.0, ...
                   1.0, 1.0, 2.0, 2.0, 2.0, ...
                   1.0, 1.0, 1.5, 1.0, 5.0, ...
                   1.0, 1.0, 1.0, 2.0, 1.0];
    new_image_roidb_train = [];
    for index = 1:numel(a_image_roidb_train)
        pseudo_boxes = check_filter_box(a_image_roidb_train(index).pseudo_boxes, lower_score);
        if (isempty(pseudo_boxes)), continue; end
        pos_boxes = {pseudo_boxes.box};   pos_boxes = cat(1, pos_boxes{:});
        pos_class = {pseudo_boxes.class}; pos_class = cat(1, pos_class{:});
        pos_score = {pseudo_boxes.score}; pos_score = cat(1, pos_score{:});
        rois        = a_image_roidb_train(index).boxes;
        rois        = [pos_boxes; rois];
        num_boxes   = size(rois, 1);
        overlap     = zeros(num_boxes, num_class, 'single');
        for bid = 1:numel(pos_class)
            gt_classes = pos_class(bid);
            gt_boxes   = pos_boxes(bid, :);
            overlap(:, gt_classes) = max(overlap(:, gt_classes), boxoverlap(rois, gt_boxes));
        end
        %append_bbox_regression_targets
        [bbox_targets, is_valid] = weakly_compute_targets(conf, rois, overlap);
        if(is_valid == false), continue; end
        new_image_roidb_train{end+1}            = a_image_roidb_train(index);
        new_image_roidb_train{end}.boxes        = rois;
        new_image_roidb_train{end}.overlap      = overlap;
        new_image_roidb_train{end}.bbox_targets = bbox_targets;
        new_image_roidb_train{end}.pseudo_boxes = pseudo_boxes;
    end
    new_image_roidb_train = cat(1, new_image_roidb_train{:});

    weakly_debug_info(conf.classes, new_image_roidb_train);

    %% Normalize targets
    num_images = length(new_image_roidb_train);
    % Infer number of classes from the number of columns in gt_overlaps
    if conf.bbox_class_agnostic
        num_classes = 1;
    else
        num_classes = size(new_image_roidb_train(1).overlap, 2);
    end
    for idx = 1:num_images
        targets = new_image_roidb_train(idx).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end), box_param.bbox_means(cls+1, :));
                new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end), box_param.bbox_stds(cls+1, :));
            end
        end
    end

end

function pseudo_boxes = check_filter_img(pseudo_boxes, image_label)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  boxes = {pseudo_boxes.box};   boxes = cat(1, boxes{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
%  if (numel(image_label) >= 3), pseudo_boxes=[]; return; end

  if (numel(setdiff(class, image_label)) > 2)
    pseudo_boxes = [];
    return;
  end

  keep = [];
  for i = 1:numel(image_label)
    cls = image_label(i);
    if (sum(class==cls) >= 4), pseudo_boxes=[]; return; end
    idx = find(class == cls);
    if (isempty(idx)), continue; end
    %[~, iii] = max(score(idx));
    %keep(end+1) = idx(iii);
    keep = [keep; idx];
  end
  pseudo_boxes = pseudo_boxes(keep);
end

function pseudo_boxes = check_filter_box(pseudo_boxes, lower_score)
   class = {pseudo_boxes.class}; class = cat(1, class{:});
   score = {pseudo_boxes.score}; score = cat(1, score{:});
   keep = [];
   for i = 1:numel(class)
     if (score(i) >= lower_score( class(i) ))
       keep(end+1) = i;
     end
   end
   pseudo_boxes = pseudo_boxes(keep);
end
