function [sampled_train, saved_train] = weakly_sample_train(image_roidb_train, num_per_class, flip)
    total = numel(image_roidb_train);
    if (flip)
        ori_ids = [];
        flip_id = [];
        for i = 1:total
            image_id = image_roidb_train(i).image_id;
            if (strcmp(image_id(end-3:end), 'flip') == 1)
                flip_id(end+1) = i;
            else
                ori_ids(end+1) = i;
            end
        end
        image_roidb_train = image_roidb_train([ori_ids,flip_id]');
        %% Check image_roidb_train 
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
            if (flip), selected_ids(end+1) = index + total; end
        end
    end
    sss_num = numel(selected_ids); if (flip), sss_num = sss_num / 2; end
    fprintf('Sample %d images (not include fliped pics)\n', sss_num);
    selected_ids  = sort(selected_ids);
    saved_ids     = setdiff((1:numel(image_roidb_train)), selected_ids);
    sampled_train = image_roidb_train(selected_ids);
    saved_train   = image_roidb_train(saved_ids);

    num_select_per_class = zeros(numel(classes), 1);
    for i = 1:numel(sampled_train)
        image_label = sampled_train(i).image_label;
        for j = 1:numel(image_label)
            num_select_per_class(image_label(j)) = num_select_per_class(image_label(j)) + 1;
        end
    end
    num_total_per_class = zeros(numel(classes), 1);
    for i = 1:numel(saved_train)
        image_label = saved_train(i).image_label;
        for j = 1:numel(image_label)
            num_total_per_class(image_label(j)) = num_total_per_class(image_label(j)) + 1;
        end
    end
    %for i = 1:numel(classes)
    %    fprintf('Class [%02d] sample : %02d / total : %5d = Limited : %2d\n', i, num_select_per_class(i), num_total_per_class(i), num_per_class(i));
    %end
end
