function [new_image_roidb_train] = weakly_full_targets(conf, image_roidb_train, bbox_means, bbox_stds)

    begin_time = tic;
    num_roidb        = numel(image_roidb_train);
    num_classes      = numel(conf.classes);

    new_image_roidb_train = [];
    for index = 1:num_roidb
        assert (isempty(image_roidb_train(index).pseudo_boxes) == false);
        pos_boxes = {image_roidb_train(index).pseudo_boxes.box};   pos_boxes = cat(1, pos_boxes{:});
        pos_class = {image_roidb_train(index).pseudo_boxes.class}; pos_class = cat(1, pos_class{:});

        rois        = image_roidb_train(index).boxes;
        rois        = [pos_boxes; rois];
        num_boxes   = size(rois, 1);
        overlap     = zeros(num_boxes, num_classes, 'single');
        for bid = 1:numel(pos_class)
            gt_classes = pos_class(bid);
            gt_boxes   = pos_boxes(bid, :);
            overlap(:, gt_classes) = max(overlap(:, gt_classes), boxoverlap(rois, gt_boxes));
        end
        %append_bbox_regression_targets
        [bbox_targets, is_valid] = weakly_compute_targets(conf, rois, overlap);
        if(is_valid == false), continue; end
        new_image_roidb_train{end+1}            = image_roidb_train(index);
        new_image_roidb_train{end}.boxes        = rois;
        new_image_roidb_train{end}.overlap      = overlap;
        new_image_roidb_train{end}.bbox_targets = bbox_targets;
    end

    new_image_roidb_train = cat(1, new_image_roidb_train{:});
    %% Normalize targets
    num_images = length(new_image_roidb_train);
    % Infer number of classes from the number of columns in gt_overlaps
    if conf.bbox_class_agnostic
        num_classes = 1;
    else
        num_classes = size(new_image_roidb_train(1).overlap, 2);
    end
    for idx = 1:num_images
        targets = new_image_roidb_train(idx).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end), bbox_means(cls+1, :));
                new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end), bbox_stds(cls+1, :));
            end
        end
    end
    fprintf('weakly_full_targets %4d -> %4d, Cost : %.1f s\n', num_roidb, num_images, toc(begin_time));
end
