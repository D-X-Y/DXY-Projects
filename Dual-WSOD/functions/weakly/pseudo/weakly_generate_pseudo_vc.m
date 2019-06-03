function [image_roidb_train] = weakly_generate_pseudo_vc(test_models, image_roidb_train)

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
    all_boxess = cell(num_roidb);
    for inet = 1:num_nets
        caffe.reset_all();
        test_net = caffe.Net( test_models(inet).test_net_def_file , 'test' );
        test_net.copy_from( test_models(inet).cur_net_file );
        conf = test_models(inet).conf;

        for index = 1:num_roidb
            if (rem(index, 1000) == 0 || index == num_roidb), fprintf('Handle %s %4d / %4d image_roidb_train, cost : %.1f s\n', test_models(inet).name, index, num_roidb, toc(begin_time)); end
            %[cur_boxes, cur_scores] = w_im_detect(conf, test_net, imread(image_roidb_train(index).image_path), image_roidb_train(index).boxes);
            [cur_boxes, cur_scores] = weakly_im_detect(conf, test_net, imread(image_roidb_train(index).image_path), image_roidb_train(index).boxes, conf.max_rois_num_in_gpu);
            %if (isempty(all_boxes{index}) && isempty(all_scores{index}))
            if (isempty(all_scores{index}))
                all_boxess{index} = cur_boxes ./ num_nets;
                all_scores{index} = cur_scores./ num_nets;
            else
                all_boxess{index} = all_boxess{index} + cur_boxes ./ num_nets;
                all_scores{index} = all_scores{index} + cur_scores./ num_nets;
            end
        end
    end

    %pseudo_boxes = cell(num_roidb, 1);
    new_bool = false(num_roidb);
    for index = 1:num_roidb
        if (rem(index, 1000) == 0 || index == num_roidb), fprintf('Handle nms %4d / %4d image_roidb_train, cost : %.1f s\n', index, num_roidb, toc(begin_time)); end
        if (rem(index, 2) == 0), reverse_idx = index - 1;
        else,                    reverse_idx = index + 1; end
        assert (check_reverse(image_roidb_train(index), image_roidb_train(reverse_idx)));

        reg_boxes = (all_boxess{index}  + reverse_box(all_boxess{reverse_idx}, image_roidb_train(reverse_idx).im_size) ) / 2;
        orign_boxes = image_roidb_train(index).boxes;
        final_score = (all_scores{index} + all_scores{reverse_idx} ) / 2;

        pseudo_boxes = generate_pseudo_v(orign_boxes, reg_boxes, final_score, num_classes, thresh_hold, image_roidb_train(index).image_label);
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

function pos_structs = generate_pseudo_v(orign_boxes, reg_boxes, final_score, num_classes, thresh_hold, image_label)
  
  %[MX_per_class, ID_cls] = max(final_score);
  %[MX_per_boxes, ID_bbx] = max(final_score, [], 2);
  %[mx_score, mx_class]   = max(MX_per_class);

  pos_structs = [];
  %for Cls = 1:num_classes
  for iii = 1:numel(image_label)

    Cls = image_label(iii);

    aboxes  = [orign_boxes, final_score(:,Cls)];
    keep           = nms(aboxes, 0.3);
    %keep           = keep(ID_bbx(keep)==Cls);
    xkeep           = keep(final_score(keep, Cls) >= thresh_hold);
    if (isempty(xkeep))
      [~, keep] = max(final_score(:, Cls));
    else
      keep = xkeep;
    end
    for j = 1:numel(keep)
      %pbox = aboxes(keep(j), 1:4);
      pbox = reg_boxes(keep(j), (Cls-1)*4+1:Cls*4);
      Struct = struct('box',   pbox, ...
                      'score', aboxes(keep(j), 5), ...
                      'class', Cls);
      pos_structs{end+1} = Struct;
    end
  end
  if (isempty(pos_structs) == false), pos_structs = cat(1, pos_structs{:}); end
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
