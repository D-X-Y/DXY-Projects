function [sampled_train, saved_train] = weakly_sample_train_v3(image_roidb_train, num_per_class, flip)
    total = numel(image_roidb_train);
    assert (flip);
    if (flip)
        ori_ids = [];
        flp_ids = [];
        for i = 1:total
            image_id = image_roidb_train(i).image_id;
            if (strcmp(image_id(end-3:end), 'flip') == 1)
                flp_ids(end+1) = i;
            else
                ori_ids(end+1) = i;
            end
        end
        image_roidb_train = image_roidb_train([ori_ids,flp_ids]');
        %% Check image_roidb_train 
        assert (numel(flp_ids) == numel(ori_ids));
        assert(rem(total, 2) == 0);
        total = total / 2;
        for i = 1:total
            image_id = image_roidb_train(i).image_id;
            flip_id  = image_roidb_train(i+total).image_id;
            assert( strcmp(flip_id, [image_id,'_flip']) == 1);
        end
        fprintf('weakly_sample_train FLIP check ok\n');
    end
    classes = cat(1, image_roidb_train.image_label);
    classes = unique(classes);
    rand_samples = randperm(total);
    selected_ids = [];
    % Select images with only one box
    for cls = 1:numel(classes)
        total_cls = [];
        for index = 1:total
          image_label = image_roidb_train(index).image_label;
          if (numel(image_label) > 1), continue; end
          if (image_label == cls)
            total_cls(end+1) = index;
          end
        end
        number = numel(total_cls);
        assert (number >= num_per_class(cls));
        total_cls = total_cls(randperm(number, num_per_class(cls)));
        selected_ids = [selected_ids; total_cls];
    end
    fprintf('Sample %d images (not include fliped pics)\n', numel(selected_ids));
    selected_ids  = sort(selected_ids);
    saved_ids     = setdiff((1:total), selected_ids);

    sampled_train = [ori_ids(selected_ids), flp_ids(selected_ids)];
    saved_train   = [ori_ids(saved_ids),    flp_ids(saved_ids)];
end
