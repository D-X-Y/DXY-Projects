function weakly_check_multibox(test_models, image_roidb_train)

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
