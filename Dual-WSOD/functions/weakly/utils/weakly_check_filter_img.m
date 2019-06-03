function ok = weakly_check_filter_img(pseudo_boxes, smallest, object_number_up)

  class = {pseudo_boxes.class}; class = cat(1, class{:});
  boxes = {pseudo_boxes.box};   boxes = cat(1, boxes{:});
  keepB = find(boxes(:,3)-boxes(:,1) >= smallest);
  keepA = find(boxes(:,4)-boxes(:,2) >= smallest);
  keep  = intersect(keepA, keepB);
  pseudo_boxes = pseudo_boxes(keep);
  class = class(keep);
  ok = [];
  %if (numel(unique(class)) >= class_number_up), return; end

  for i = 1:numel(class)
    if (numel(find(class == class(i))) >= object_number_up), return; end
  end
  ok = pseudo_boxes;

end
