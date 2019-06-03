function pseudo_boxes = weakly_nms_max_pseudo(pseudo_boxes, min_thresh)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  boxes = {pseudo_boxes.box};   boxes = cat(1, boxes{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
  unique_cls = unique(class);

  keep = false(numel(class), 1);
  for c = 1:numel(unique_cls)
    cls = unique_cls(c);
    idx = find(class == cls);
    curkeep = nms_max([boxes(idx,:), score(idx,:)], min_thresh);
    keep( idx(curkeep) ) = true;
  end
  pseudo_boxes = pseudo_boxes(keep);

end
