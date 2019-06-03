function [image_roidb_train] = weakly_filter_score(test_models, image_roidb_train, SAVE_TERM)

  classes = test_models{1}.conf.classes;
  num = numel(image_roidb_train); num_class = numel(classes);
  oks = false(num);               begin_time = tic;
  SAVE_TERM = ceil(SAVE_TERM);
  %% multibox_thresh = 0;
  %% Filter Multiple Boxes
  lower_score = cell(num_class,1);
  for idx = 1:num
    pseudo_boxes = check_filter_img(image_roidb_train(idx).pseudo_boxes);
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
  fprintf('weakly_filter_score[check] after filter left %4d images\n', numel(image_roidb_train));

  for cls = 1:num_class
    scores = lower_score{cls};
    if (isempty(scores))
      lower_score{cls} = 0;
    else
      [scorted_score, ~] = sort(scores, 'descend');
      lower_score{cls} = scorted_score(min(end, SAVE_TERM(cls)));
    end
  end
  lower_score = cat(1, lower_score{:});

  oks = false(numel(image_roidb_train), 1);
  for idx = 1:numel(image_roidb_train)
    pseudo_boxes = check_filter_score(image_roidb_train(idx).pseudo_boxes, lower_score);
    if (isempty(pseudo_boxes)), continue; end
    
    oks(idx) = true;
    image_roidb_train(idx).pseudo_boxes = pseudo_boxes;
  end

  image_roidb_train = image_roidb_train(oks);
  weakly_debug_info( classes, image_roidb_train );
  fprintf('weakly_filter_score[final] after filter left %4d images\n', numel(image_roidb_train));
end

function pseudo_boxes = check_filter_score(pseudo_boxes, lower_score)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
  boxes = {pseudo_boxes.box};   boxes = cat(1, boxes{:});
  keep = false(numel(pseudo_boxes), 1);
  for i = 1:numel(class)
    if ( boxes(i,3)-boxes(i,1) <= 15 ), continue; end
    if ( boxes(i,4)-boxes(i,2) <= 15 ), continue; end
    if (score(i) >= lower_score( class(i) ))
      keep(i) = true;
    end
  end
  pseudo_boxes = pseudo_boxes(keep);
end

function ok = check_filter_img(pseudo_boxes)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  ok = [];
  if (numel(unique(class)) >= 5), return; end
  for i = 1:numel(class)
    if (numel(find(class == class(i))) >= 4), return; end
  end
  ok = pseudo_boxes;
end

function ok = check_multibox(conf, test_nets, roidb_train, thresh)
  max_rois_num_in_gpu = 10000;
  boxes = {roidb_train.pseudo_boxes.box}; boxes = cat(1, boxes{:});
  cls   = {roidb_train.pseudo_boxes.class}; cls = cat(1, cls{:});
  ori_score = {roidb_train.pseudo_boxes.score}; ori_score = cat(1, ori_score{:});
  num_boxes = size(boxes, 1);
  Fboxes = []; Fscores = [];
  for j = 1:numel(test_nets)
    [Tboxes, Tscores] = weakly_im_detect(conf, test_nets{j}, imread(roidb_train.image_path), multibox(boxes), max_rois_num_in_gpu);
    if (j == 1)
      Fboxes = Tboxes; Fscores = Tscores;
    else
      Fboxes = Fboxes + Tboxes; Fscores = Fscores + Tscores;
    end
  end
  Fboxes = Fboxes / numel(test_nets);
  Fscores = Fscores / numel(test_nets);
  ok = true(num_boxes, 1);
  for idx = 1:4
    cscores = Tscores((idx-1)*num_boxes+1 : idx*num_boxes, :);
    [mx_score, mx_cls] = max(cscores, [], 2);
    for j = 1:numel(cls)
      if(mx_cls(j) == cls(j) && mx_score(j)+thresh > ori_score(j))
        ok(j) = false;
      end
    end
  end
end


function boxes = multibox(boxes)
  ANS = zeros(0, 4);
  CUR = boxes; CUR(:,4) = (CUR(:,2)+CUR(:,4)) / 2;
  ANS = [ANS;CUR];
  CUR = boxes; CUR(:,2) = (CUR(:,2)+CUR(:,4)) / 2;
  ANS = [ANS;CUR];
  CUR = boxes; CUR(:,3) = (CUR(:,1)+CUR(:,3)) / 2;
  ANS = [ANS;CUR];
  CUR = boxes; CUR(:,1) = (CUR(:,1)+CUR(:,3)) / 2;
  ANS = [ANS;CUR];
  boxes = ANS;
end
