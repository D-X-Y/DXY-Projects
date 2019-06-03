function weakly_debug_info(classes, image_roidb_train, losses, loss_threshes)
    begin_time = tic;
    num_image = numel(image_roidb_train);
    num_class = numel(classes);
    
    if ~exist('losses', 'var')
      losses = inf(num_image, num_class);
    end
    if ~exist('loss_threshes', 'var')
      loss_threshes = inf(num_class, 1);
    end

    total_boxes = zeros(num_class, 1);
    todet_boxes = zeros(num_class, 1);
    cordt_boxes = zeros(num_class, 1);
    truee_boxes = zeros(num_class, 1);

    total_image = zeros(num_class, 1);
    todet_image = zeros(num_class, 1);
    corrt_image = zeros(num_class, 1);
    truee_image = zeros(num_class, 1);

    lower_score = inf(num_class, 1);
    hight_score = zeros(num_class, 1);
    lower_loss  = inf(num_class, 1);
    hight_loss  = zeros(num_class, 1);

    for j = 1:num_image
      for cls = 1:num_class
        [dtboxes, dtscore] = inloop_g2box(image_roidb_train(j).pseudo_boxes, cls);
        gtboxes = image_roidb_train(j).Debug_GT_Box( image_roidb_train(j).Debug_GT_Cls == cls, :);
        if (isempty(dtboxes) && isempty(gtboxes)), continue; end
        total_boxes(cls) = total_boxes(cls) + size(gtboxes, 1);
        todet_boxes(cls) = todet_boxes(cls) + size(dtboxes, 1);
        for x = 1:size(gtboxes, 1)
            overlap = iou(gtboxes(x,:), dtboxes);
            if (max(overlap) >= 0.5), cordt_boxes(cls) = cordt_boxes(cls) + 1; end
        end
        for x = 1:size(dtboxes, 1)
            overlap = iou(dtboxes(x,:), gtboxes);
            if (max(overlap) >= 0.5), truee_boxes(cls) = truee_boxes(cls) + 1; end
            lower_score(cls) = min(lower_score(cls), dtscore(x));
            hight_score(cls) = max(hight_score(cls), dtscore(x));
        end
        lower_loss(cls) = min(lower_loss(cls), losses(j, cls));
        if (losses(j, cls) <= 10) % avoid inf
            hight_loss(cls) = max(hight_loss(cls), losses(j, cls));
        end
      end
      image_label = image_roidb_train(j).image_label;

      total_image(image_label) = total_image(image_label) + 1;
      det_label = {image_roidb_train(j).pseudo_boxes.class};
      det_label = cat(1, det_label{:});
      todet_image(det_label) = todet_image(det_label) + 1;

      cor_label = intersect(image_label, det_label);
      corrt_image(cor_label) = corrt_image(cor_label) + 1;
      tue_label = intersect(det_label, image_label);
      truee_image(tue_label) = truee_image(tue_label) + 1;
    end

    for Cls = 1:num_class
    fprintf('--[%02d][%12s] B=[prec: %4d/%4d = %.3f ; rec: %4d/%4d = %.3f] I=[prec: %4d/%4d = %.3f ; rec: %4d/%4d = %.3f] S=[%.3f, %.3f] L=[%.3f, %.3f](thresh: %.3f)\n', Cls, classes{Cls}, ...
                 truee_boxes(Cls), todet_boxes(Cls), divide( truee_boxes(Cls), todet_boxes(Cls) ), ...
                 cordt_boxes(Cls), total_boxes(Cls), divide( cordt_boxes(Cls), total_boxes(Cls) ), ...
                 truee_image(Cls), todet_image(Cls), divide( truee_image(Cls), todet_image(Cls) ), ...
                 corrt_image(Cls), total_image(Cls), divide( corrt_image(Cls), total_image(Cls) ), ...
                 lower_score(Cls), hight_score(Cls), lower_loss(Cls), hight_loss(Cls), loss_threshes(Cls));
    end
    
    fprintf('Debug_Info %4d : B=[prec: %4d/%4d = (%.3f,%.3f) ; rec: %4d/%4d = (%.3f,%.3f)] I=[prec:%4d/%4d = (%.3f,%.3f) ; rec: %4d/%4d = (%.3f,%.3f)] cost : %.1f s \n', num_image, ...
                 sum(truee_boxes), sum(todet_boxes), divide( sum(truee_boxes), sum(todet_boxes) ), divide( truee_boxes, todet_boxes ), ...
                 sum(cordt_boxes), sum(total_boxes), divide( sum(cordt_boxes), sum(total_boxes) ), divide( cordt_boxes, total_boxes ), ...
                 sum(truee_image), sum(todet_image), divide( sum(truee_image), sum(todet_image) ), divide( truee_image, todet_image ), ...
                 sum(corrt_image), sum(total_image), divide( sum(corrt_image), sum(total_image) ), divide( corrt_image, total_image ), toc(begin_time));
                 
end

function [total] = divide(x, y)
  assert (numel(x) == numel(y));
  total = zeros(numel(x), 1);
  for i = 1:numel(x)
    if (y(i) ~= 0), total(i) = x(i) / y(i); end
  end
  total = mean(total);
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
  
