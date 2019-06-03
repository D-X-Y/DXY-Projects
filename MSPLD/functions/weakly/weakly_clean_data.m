function [image_roidb_train] = weakly_clean_data(classes, image_roidb_train, smallest)

  num = numel(image_roidb_train); num_class = numel(classes);
  oks = false(num);               begin_time = tic;
  %% Filter Multiple Boxes
  lower_score = cell(num_class,1);
  class_number_up = 4;
  object_number_up = 4;
  for idx = 1:num
    pseudo_boxes = weakly_nms_max_pseudo(image_roidb_train(idx).pseudo_boxes, 0.3);
    if (isempty(pseudo_boxes)), continue; end
    pseudo_boxes = weakly_check_filter_img(image_roidb_train(idx).pseudo_boxes, smallest, class_number_up, object_number_up);
    if (isempty(pseudo_boxes)), continue; end

    image_roidb_train(idx).pseudo_boxes = pseudo_boxes;
    oks(idx) = true;
    for j = 1:numel(pseudo_boxes)
        class = pseudo_boxes(j).class;
        score = pseudo_boxes(j).score;
        lower_score{class}(end+1) = score;
    end
  end

  image_roidb_train = image_roidb_train(oks);
  fprintf('weakly_clean_data after filter left %4d images\n', numel(image_roidb_train));

  weakly_debug_info( classes, image_roidb_train );
end
