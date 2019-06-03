function mAP = weakly_co_test_mAP(confs, imdb, roidb, varargin)
% --------------------------------------------------------
% MSPLD implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('confs',                             @iscell);
    ip.addRequired('imdb',                              @isstruct);
    ip.addRequired('roidb',                             @isstruct);
    ip.addParamValue('test_iteration',  1,              @isscalar);
    ip.addParamValue('net_defs',                        @iscell);
    ip.addParamValue('net_models',                      @iscell);
    ip.addParamValue('cache_name',      '',             @isstr);
    ip.addParamValue('rng_seed',         5,             @isscalar);
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('log_prefix',      '',             @isstr);
    ip.addParamValue('dis_itertion',    500,            @isscalar);
    ip.addParamValue('ignore_cache',    false,          @islogical);
    
    ip.parse(confs, imdb, roidb, varargin{:});
    opts = ip.Results;
    

    assert (numel(opts.net_defs) == numel(opts.net_models));
    assert (numel(opts.net_defs) == numel(confs));
    weakly_assert_conf(confs);
    classes = confs{1}.classes;
%%  set cache dir
    cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, [imdb.name, '_mAP']);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', [opts.log_prefix, 'test_', timestamp, '.txt']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    num_classes = imdb.num_classes;
    caffe.reset_all(); 
    
    try
      aboxes = cell(num_classes, 1);
      if opts.ignore_cache
          throw('');
      end
      for i = 1:num_classes
        load(fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]));
        aboxes{i} = boxes;
      end
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = [];
        for i = 1:numel(opts.net_defs)
          caffe_net{i} = caffe.Net(opts.net_defs{i}, 'test');
          caffe_net{i}.copy_from(opts.net_models{i});
        end

        % set random seed
        prev_rng = seed_rand(opts.rng_seed);
        caffe.set_random_seed(opts.rng_seed);

        % set gpu/cpu
        if confs{1}.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 

        disp('opts:');
        disp(opts);
        for i = 1:numel(confs)
            fprintf('conf : %d :', i);
            disp(confs{i});
        end
        
        %heuristic: keep an average of 160 detections per class per images prior to NMS
        max_per_set = 160 * num_images;
        % heuristic: keep at most 400 detection per class per image prior to NMS
        max_per_image = 400;
        % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
        thresh = -inf * ones(num_classes, 1);
        % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = cell(num_classes, 1);
        % all detections are collected into:
        %    all_boxes[cls][image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
        aboxes = cell(num_classes, 1);
        box_inds = cell(num_classes, 1);
        for i = 1:num_classes
            aboxes{i} = cell(length(imdb.image_ids), 1);
            box_inds{i} = cell(length(imdb.image_ids), 1);
        end

        count = 0;
        t_start = tic;
        for i = 1:num_images
            count = count + 1;
            if (rem(count, opts.dis_itertion) == 1)
              fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            end
            th = tic;
            d = roidb.rois(i);
            im = imread(imdb.image_at(i));
            
            boxes  = [];
            scores = [];
            for jj = 1:numel(caffe_net)
                pre_boxes = d.boxes(~d.gt, :);
                [cboxes, cscores] = weakly_im_detect(confs{jj}, caffe_net{jj}, im, pre_boxes, confs{jj}.max_rois_num_in_gpu);
				if (opts.test_iteration == 2)
					[~, mx_id] = max(cscores, [], 2);
					mx_id = (mx_id-1)*4;
					add_boxes = single(zeros(size(pre_boxes)));
					parfor box_id = 1:size(pre_boxes,1)
						for coor = 1:4, add_boxes(box_id, coor) = cboxes(box_id, mx_id(box_id)+coor); end
					end
					pre_boxes = (pre_boxes+add_boxes)./2;
					[cboxes, cscores] = weakly_im_detect(confs{jj}, caffe_net{jj}, im, pre_boxes, confs{jj}.max_rois_num_in_gpu);
				else
					assert(opts.test_iteration == 1);
				end

                if (isempty(boxes) && isempty(scores))
                    boxes  = cboxes;
                    scores = cscores;
                else
                    boxes  = boxes + cboxes;
                    scores = scores + cscores;
                end
            end
            boxes  = boxes  ./ numel(caffe_net);
            scores = scores ./ numel(caffe_net);

            for j = 1:num_classes
                inds = find(scores(:, j) > thresh(j));
                if ~isempty(inds)
                    [~, ord] = sort(scores(inds, j), 'descend');
                    ord = ord(1:min(length(ord), max_per_image));
                    inds = inds(ord);
                    cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
                    cls_scores = scores(inds, j);
                    aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
                    box_inds{j}{i} = [box_inds{j}{i}; inds];
                else
                    aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
                    box_inds{j}{i} = box_inds{j}{i};
                end
            end
            
            if (rem(count, opts.dis_itertion) == 1)
              fprintf(' time %.3fs\n', toc(th)); 
            end
            if mod(count, 1000) == 0
                for j = 1:num_classes
                [aboxes{j}, box_inds{j}, thresh(j)] = ...
                    keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
                end
                %disp(thresh);
                diary; diary;
            end    
        end

        for j = 1:num_classes
            [aboxes{j}, box_inds{j}, thresh(j)] = ...
                keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
        end
        disp(thresh');

        for i = 1:num_classes
            top_scores{i} = sort(top_scores{i}, 'descend');  
            if (length(top_scores{i}) > max_per_set)
                thresh(i) = top_scores{i}(max_per_set);
            end

            % go back through and prune out detections below the found threshold
            for j = 1:length(imdb.image_ids)
                if ~isempty(aboxes{i}{j})
                    I = find(aboxes{i}{j}(:,end) < thresh(i));
                    aboxes{i}{j}(I,:) = [];
                    box_inds{i}{j}(I,:) = [];
                end
            end

            save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]);
            boxes = aboxes{i};
            inds = box_inds{i};
            save(save_file, 'boxes', 'inds');
            clear boxes inds;
        end
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end

    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------

	tic;
    if isequal(imdb.eval_func, @imdb_eval_voc)
        %new_parpool();
        parfor model_ind = 1:num_classes
          cls = imdb.classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, opts.cache_name, opts.suffix);
        end
    else
    % ilsvrc
        res = imdb.eval_func(aboxes, imdb, opts.cache_name, opts.suffix);
    end

    if ~isempty(res)
        fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
        fprintf('Results:\n');
        aps = [res(:).ap]' * 100;
        %disp(aps);
        assert( numel(classes) == numel(aps));
        for idx = 1:numel(aps)
            fprintf('%12s : %5.2f\n', classes{idx}, aps(idx));
        end
        fprintf('\nmean mAP : %.4f\n', mean(aps));
        %disp(mean(aps));
        fprintf('~~~~~~~~~~~~~~~~~~~~ evaluate cost %.2f s\n', toc);
        mAP = mean(aps);
    else
        mAP = nan;
    end
    
    diary off;
end


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{1:end_at});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:end_at
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,end) >= thresh);
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end
