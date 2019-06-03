function mAP = weakly_test_mAP_v2(confs, imdb, roidb, varargin)
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
    
%%  testing 
    % init caffe net
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    num_nets = numel(opts.net_defs);
    %caffe_net = [];
    %for i = 1:numel(opts.net_defs)
    %  caffe_net{i} = caffe.Net(opts.net_defs{i}, 'test');
    %  caffe_net{i}.copy_from(opts.net_models{i});
    %end

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
    % all detections are collected into:
    %    all_boxes[cls][image] = N x 5 array of detections in
    %    (x1, y1, x2, y2, score)
    aboxes = cell(num_classes, 1);
    for i = 1:num_classes
      aboxes{i} = cell(length(imdb.image_ids), 1);
    end

    t_start = tic;
    for inet = 1:num_nets
      count = 0;
      caffe.reset_all();
      fprintf('>>> %3d/%3d net procedure : %s \n', inet, num_nets, opts.net_defs{inet});
      caffe_net = caffe.Net(opts.net_defs{inet}, 'test');
      caffe_net.copy_from(opts.net_models{inet});

      for i = 1:num_images
        count = count + 1;
        if (rem(count, opts.dis_itertion) == 1), fprintf('%s: test (%s) %d/%d cost : %.1f s\n', procid(), imdb.name, count, num_images, toc(t_start)); end
        d = roidb.rois(i);
        im = imread(imdb.image_at(i));

        [boxes, scores] = weakly_im_detect(confs{inet}, caffe_net, im, d.boxes(~d.gt,:), confs{inet}.max_rois_num_in_gpu);
        for j = 1:num_classes
            cls_boxes = boxes(:, (1+(j-1)*4):((j)*4));
            cls_scores = scores(:, j);
            temp = cat(2, single(cls_boxes), single(cls_scores)) / num_nets;
            if (isempty(aboxes{j}{i}))
              aboxes{j}{i} = temp;
            else
              aboxes{j}{i} = aboxes{j}{i} + temp;
            end
        end
      end
    end

    for j = 1:num_classes
      [aboxes{j}, thresh(j)] = ...
        keep_top_k(aboxes{j}, max_per_set, thresh(j));
    end
    disp(thresh');

    fprintf('test all images in %f seconds.\n', toc(t_start));
        
    caffe.reset_all(); 
    rng(prev_rng);

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
function [boxes, thresh] = keep_top_k(boxes, top_k, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{:});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:numel(boxes)
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,end) >= thresh);
            boxes{image_index} = bbox(keep,:);
        end
    end
end
