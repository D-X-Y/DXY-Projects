function [sampled_train, saved_train] = weakly_sample_train_v2(image_roidb_train, num_per_class, flip)
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
    num_select_per_class = zeros(numel(classes), 1);
    rand_samples = randperm(total);
    selected_ids = [];
    num_class = zeros(numel(classes),1);
    for index = 1:total
        num_class(index) = numel(image_roidb_train(index).image_label);
    end
    %[~, rand_samples] = sort(num_class, 'descend');
    %[~, rand_samples] = sort(num_class, 'descend');
    for i = 1:total
        index  = rand_samples(i);
        number = numel(image_roidb_train(index).image_label);
        label  = image_roidb_train(index).image_label(randperm(number,1));
        if (num_select_per_class(label) < num_per_class(label))
	        num_select_per_class(label) = num_select_per_class(label) + 1;
	        selected_ids(end+1) = index;
            %if (flip), selected_ids(end+1) = index + total; end
        end
    end
    fprintf('Sample %d images (not include fliped pics)\n', numel(selected_ids));
    selected_ids  = sort(selected_ids);
    saved_ids     = setdiff((1:total), selected_ids);

    sampled_train = [ori_ids(selected_ids), flp_ids(selected_ids)];
    saved_train   = [ori_ids(saved_ids),    flp_ids(saved_ids)];
end
