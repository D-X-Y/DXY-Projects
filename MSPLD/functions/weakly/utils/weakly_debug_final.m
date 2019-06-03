function weakly_debug_final(classes, debug_cache_dir, image_roidb)
  pseudo_boxes = image_roidb.pseudo_boxes;
  if (isempty(pseudo_boxes)), return; end
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
  boxes = {pseudo_boxes.box}  ; boxes = cat(1, boxes{:});
  assert (size(boxes,1) == size(class,1) && size(class,1) == size(score,1));
  boxes_cell = cell(length(classes), 1);
  for i = 1:length(classes)
    boxes_cell{i} = zeros(0,5);
  end
  grt_boxes_cell = boxes_cell;
  for i = 1:length(class)
    boxes_cell{class(i)} = [boxes_cell{class(i)}; [boxes(i,:), score(i)]];
  end
  mkdir_if_missing(debug_cache_dir);
  im = imread(image_roidb.image_path);
  showboxes(im, boxes_cell, classes, 'voc');
  split_name = image_roidb.image_id;
  saveas(gcf, fullfile(debug_cache_dir, [split_name, '.jpg']));
  
%%% Draw Ground Truth
  for i = 1:length(image_roidb.Debug_GT_Cls)
    grt_boxes_cell{image_roidb.Debug_GT_Cls(i)} = [grt_boxes_cell{image_roidb.Debug_GT_Cls(i)}; [image_roidb.Debug_GT_Box(i,:), 1]];
  end
  showboxes(im, grt_boxes_cell, classes, 'voc');
  saveas(gcf, fullfile(debug_cache_dir, [split_name, '_ok.jpg']));
end
