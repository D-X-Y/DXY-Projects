function [pred_boxes, scores] = weakly_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% --------------------------------------------------------
% MSPLD implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    assert (numel(caffe_net.inputs) == 2);
    regression = conf.regression;
    if (regression)
        assert (numel(caffe_net.outputs) == 2);
    else
        assert (numel(caffe_net.outputs) == 1);
    end
    
    [im_blob, rois_blob, ~] = get_blobs(conf, im, boxes);
    
    % When mapping from image ROIs to feature map ROIs, there's some aliasing
    % (some distinct image ROIs get mapped to the same feature ROI).
    % Here, we identify duplicate feature ROIs, so we only compute features
    % on the unique subset.
    [~, index, inv_index] = unique(rois_blob, 'rows');
    rois_blob = rois_blob(index, :);
    boxes = boxes(index, :);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);
    rois_blob = single(rois_blob);
    
    total_rois = size(rois_blob, 4);
    total_scores = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    if (regression)
        total_box_deltas = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    end
    for i = 1:ceil(total_rois / max_rois_num_in_gpu)
        
        sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
        sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
        sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
        
        net_inputs = {im_blob, sub_rois_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        caffe_net.forward(net_inputs);

        if conf.test_binary
            % simulate binary logistic regression
            scores = caffe_net.blobs('cls_score').get_data();
            scores = squeeze(scores)';
            % Return scores as fg - bg
            scores = bsxfun(@minus, scores, scores(:, 1));
        else
            % use softmax estimated probabilities
            scores = caffe_net.blobs('cls_prob').get_data();
            scores = squeeze(scores)';
        end

        % Apply bounding-box regression deltas
        if (regression)
            box_deltas = caffe_net.blobs('bbox_pred').get_data();
            box_deltas = squeeze(box_deltas)';
            total_box_deltas{i} = box_deltas;
        end
        total_scores{i} = scores;
    end 
    
    scores = cell2mat(total_scores);

    if (regression)
        box_deltas = cell2mat(total_box_deltas);

        pred_boxes = rfcn_bbox_transform_inv(boxes, box_deltas);
        pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));
    end

    % Map scores and predictions back to the original set of boxes
    scores = scores(inv_index, :);
    if (regression)
        pred_boxes = pred_boxes(inv_index, :);
    end
    % remove scores and boxes for back-ground
    if (regression)
        pred_boxes = pred_boxes(:, 5:end);
    end
    scores = scores(:, 2:end);
    if (conf.bbox_class_agnostic && regression)
        pred_boxes = repmat(pred_boxes, [1, size(scores,2)]);
    end
    if (regression == false)
        assert (size(boxes, 1) == size(scores, 1));
        pred_boxes = repmat(boxes(inv_index, :), 1, size(scores, 2));
    end
end

function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function [blob, im_scales] = get_image_blob(conf, im)
    [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales);
    blob = im_list_to_blob(ims);    
end

function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
    [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
    rois_blob = single([levels, feat_rois]);
end

function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)
    im_rois = single(im_rois);
    
    if length(scales) > 1
        widths = im_rois(:, 3) - im_rois(:, 1) + 1;
        heights = im_rois(:, 4) - im_rois(:, 2) + 1;
        
        areas = widths .* heights;
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
        [~, levels] = min(abs(scaled_areas - 224.^2), [], 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1;
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end
