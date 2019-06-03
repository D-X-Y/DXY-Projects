function [image_roidb_train] = weakly_generate_pseudo(test_models, image_roidb_train, boost)
% --------------------------------------------------------
% MSPLD implementation
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    begin_time = tic;
    test_models      = cat(1, test_models{:});
    weakly_assert_conf( {test_models.conf} );
    classes = test_models(1).conf.classes;
    num_roidb        = numel(image_roidb_train);       assert (rem(num_roidb,2) == 0);
    num_classes      = numel(classes);
    num_nets         = numel(test_models);
    thresh_hold = 0.2;

    for index = 1:num_roidb
        assert (isempty(image_roidb_train(index).overlap));
        assert (isempty(image_roidb_train(index).bbox_targets));
    end
    %all_boxes  = cell(num_roidb);
    all_scores = cell(num_roidb);
    for inet = 1:num_nets
        caffe.reset_all();
        test_net = caffe.Net( test_models(inet).test_net_def_file , 'test' );
        test_net.copy_from( test_models(inet).cur_net_file );
        conf = test_models(inet).conf;

        for index = 1:num_roidb
            if (rem(index, 2000) == 0 || index == num_roidb), fprintf('Handle %s %4d / %4d image_roidb_train, cost : %.1f s\n', test_models(inet).name, index, num_roidb, toc(begin_time)); end
            %[cur_boxes, cur_scores] = w_im_detect(conf, test_net, imread(image_roidb_train(index).image_path), image_roidb_train(index).boxes, boost);
            [~, cur_scores] = w_im_detect(conf, test_net, imread(image_roidb_train(index).image_path), image_roidb_train(index).boxes, boost);
            %if (isempty(all_boxes{index}) && isempty(all_scores{index}))
            if (isempty(all_scores{index}))
            %    all_boxes{index}  = cur_boxes ./ num_nets;
                all_scores{index} = cur_scores./ num_nets;
            else
            %    all_boxes{index}  = all_boxes{index} + cur_boxes ./ num_nets;
                all_scores{index} = all_scores{index}+ cur_scores./ num_nets;
            end
        end
    end

    %pseudo_boxes = cell(num_roidb, 1);
    new_bool = false(num_roidb);
    for index = 1:num_roidb
        if (rem(index, 1000) == 0 || index == num_roidb), fprintf('Handle nms %4d / %4d image_roidb_train, cost : %.1f s\n', index, num_roidb, toc(begin_time)); end
        reverse_idx = index + (num_roidb/2);
        if (reverse_idx > num_roidb), reverse_idx = reverse_idx - num_roidb; end
        assert (check_reverse(image_roidb_train(index), image_roidb_train(reverse_idx)));

        %final_boxes = (all_boxes{index}  + reverse_box(all_boxes{reverse_idx}, image_roidb_train(reverse_idx).im_size) ) / 2;
        final_boxes = image_roidb_train(index).boxes;
        final_score = (all_scores{index} + all_scores{reverse_idx} ) / 2;

        pseudo_boxes = generate_pseudo(final_boxes, final_score, num_classes, thresh_hold);
        if (isempty(pseudo_boxes) == false)
            new_bool (index) = true;
            image_roidb_train(index).pseudo_boxes = pseudo_boxes;
        end
    end

    image_roidb_train = image_roidb_train( new_bool );
    weakly_debug_info( classes, image_roidb_train );
    fprintf('Generate new_image_roidb_train : %4d -> %4d, Cost : %.1f s\n', num_roidb, numel(image_roidb_train), toc(begin_time));
    caffe.reset_all();
end

function pos_structs = generate_pseudo(final_boxes, final_score, num_classes, thresh_hold)
  
  if (size(final_boxes, 2) == num_classes*4)
    single = false;
  else
    single = true;
  end
  [MX_per_class, ID_cls] = max(final_score);
  [MX_per_boxes, ID_bbx] = max(final_score, [], 2);
  [mx_score, mx_class]   = max(MX_per_class);

  pos_structs = [];
  for Cls = 1:num_classes
    if (single == false)
      aboxes  = [final_boxes(:,(Cls-1)*4+1:Cls*4), final_score(:,Cls)];
    else
      aboxes  = [final_boxes, final_score(:,Cls)];
    end
    keep           = nms(aboxes, 0.3);
    keep           = keep(ID_bbx(keep)==Cls);
    keep           = keep(final_score(keep, Cls) >= thresh_hold);
    for j = 1:numel(keep)
      pbox = aboxes(keep(j), 1:4);
      Struct = struct('box',   pbox, ...
                      'score', aboxes(keep(j), 5), ...
                      'class', Cls);
      pos_structs{end+1} = Struct;
    end
  end
  if (isempty(pos_structs) == false), pos_structs = cat(1, pos_structs{:}); end
end

function [boxes, scores] = w_im_detect(conf, test_net, image, in_boxes, boost)

  [boxes, scores] = weakly_im_detect(conf, test_net, image, in_boxes, conf.max_rois_num_in_gpu);

  if (boost)
    [~, mx_id] = max(scores, [], 2);
    mx_id = (mx_id-1)*4;
    add_boxes = single(zeros(size(in_boxes)));
    parfor box_id = 1:size(in_boxes,1)
        for coor = 1:4, add_boxes(box_id, coor) = boxes(box_id, mx_id(box_id)+coor); end
    end
    in_boxes = (in_boxes + add_boxes) ./ 2;
    [boxes, scores] = weakly_im_detect(conf, test_net, image, in_boxes, conf.max_rois_num_in_gpu);
  end

end

function rev_box = reverse_box(boxes, im_size)
   rev_box = boxes;
   rev_box(:, [1,3]) = im_size(2) + 1 - rev_box(:, [3,1]);
end

function [ok] = check_reverse(A, B)
   ok = (all(A.im_size == B.im_size));
   if (ok == false), return; end
   ok = (all(all(reverse_box(A.boxes, A.im_size) == B.boxes)));
end
