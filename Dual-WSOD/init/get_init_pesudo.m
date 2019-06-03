function new_image_roidb_train = get_init_pesudo(conf, image_roidb_train, nms_thresh, filter_thresh)
    num_image = numel(image_roidb_train);
    num_class = numel(conf.classes);
    new_image_roidb_train = [];
    tic;
    for idx = 1:num_image
        pseudo_boxes = inloop_get_pesudo(conf, image_roidb_train(idx), image_roidb_train(idx).corloc, nms_thresh, filter_thresh);
        if (isempty(pseudo_boxes)), continue; end
        new_image_roidb_train{end+1} = image_roidb_train(idx);
        new_image_roidb_train{end}.pseudo_boxes = pseudo_boxes;
        if (rem(idx, 1500) == 0 || idx == num_image)
            fprintf('--init-handle %4d / %4d image_roidb_train, cost : %.1f s\n', idx, num_image, toc);
        end
    end
    new_image_roidb_train = cat(1, new_image_roidb_train{:});

    weakly_debug_info(conf.classes, new_image_roidb_train);
    fprintf('Init get_init_pesudo done, cost %.1f s\n', toc);
end

function [score, class] = inloop_extract(image_roidb_train)
  class = {image_roidb_train.pseudo_boxes.class};
  class = cat(1, class{:});
  score = {image_roidb_train.pseudo_boxes.score};
  score = cat(1, score{:});
end
  

function [boxes, score] = inloop_g2box(pseudo_boxes, cls)
  class = {pseudo_boxes.class};
  class = cat(1, class{:});
  score = {pseudo_boxes.score};
  score = cat(1, score{:});
  boxes = {pseudo_boxes.box};
  boxes = cat(1, boxes{:});
  boxes = boxes(class == cls, :);
  score = score(class == cls);
end
  

function structs = inloop_get_pesudo(conf, image_roidb_train, corloc_info, nms_thresh, filter_thresh)
  num_class = size(corloc_info.score, 2);
  assert (num_class == numel(conf.classes));
  %for cls = 1:num_class
  structs = [];
  %for j = 1:numel(image_roidb_train.image_label)
  %  cls = image_roidb_train.image_label(j);
  for cls = 1:num_class
    aboxes = [corloc_info.rois, corloc_info.score(:, cls)];
    keep = nms(aboxes, nms_thresh);
    keep = keep(corloc_info.score(keep, cls) >= filter_thresh);
    for jj = 1:numel(keep)
      Struct = struct('box',   aboxes(keep(jj), 1:4), ...
                      'score', aboxes(keep(jj), 5), ...
                      'class', cls);
      structs{end+1} = Struct;
    end
  end
  if (isempty(structs) == false), structs = cat(1, structs{:}); end
end
