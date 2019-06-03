function pseudo_boxes = weakly_nms_max_pseudo(pseudo_boxes, image_label, min_thresh)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  boxes = {pseudo_boxes.box};   boxes = cat(1, boxes{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
  %unique_cls = unique(class);

  keep = false(numel(class), 1);
  %for c = 1:numel(unique_cls)
  for c = 1:numel(image_label)
    cls = image_label(c);
    idx = find(class == cls);
    if (isempty(idx)), continue; end
    curkeep = nms_max([boxes(idx,:), score(idx,:)], min_thresh);
    keep( idx(curkeep) ) = true;
  end
  pseudo_boxes = pseudo_boxes(keep);

end
