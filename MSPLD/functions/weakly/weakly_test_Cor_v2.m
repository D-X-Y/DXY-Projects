function mean_loc = weakly_test_Cor_v2(confs, imdb, roidb, varargin)
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
    ip.addParamValue('net_defs',                        @iscell);
    ip.addParamValue('net_models',                      @iscell);
    ip.addParamValue('cache_name',      '',             @isstr);
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('rng_seed',         5,             @isscalar);
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
    cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, [imdb.name, '_Cor']);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', [opts.log_prefix, 'test_', timestamp, '.txt']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    num_classes = imdb.num_classes;
    caffe.reset_all(); 
    
    gt_boxes = cell(num_images, 1);
    for i = 1:num_images
      gt = roidb.rois(i).gt;
      Struct = struct('class', roidb.rois(i).class(gt), ...
                      'boxes', roidb.rois(i).boxes(gt,:));
      gt_boxes{i} = Struct;
    end
    
%%  testing 
    % init caffe net
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    %caffe_net = [];
    %for i = 1:numel(opts.net_defs)
    %    caffe_net{i} = caffe.Net(opts.net_defs{i}, 'test');
    %    caffe_net{i}.copy_from(opts.net_models{i});
    %end
    num_nets = numel(opts.net_defs);

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
    % all detections are collected into:
    %    all_boxes[image] = 20 x 4 array of detections in
    %    (x1, y1, x2, y2) for each class
    all_boxes = cell(num_images, num_classes);

    t_start = tic;
    for inet = 1:num_nets
      count = 0;
      caffe.reset_all();
      fprintf('>>> %3d/%3d net procedure %s \n', inet, num_nets, opts.net_defs{inet});
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
            if (isempty(all_boxes{i,j}))
              all_boxes{i,j} = temp;
            else
              all_boxes{i,j} = all_boxes{i,j} + temp;
            end
        end
      end
    end

    for i = 1:num_images
      for cls = 1:num_classes
        tscore = all_boxes{i,cls}(:,end);
        tboxes = all_boxes{i,cls}(:,1:4);
        [~, idx] = max(tscore);
        all_boxes{i,cls} = tboxes(idx,:);
      end
    end

    caffe.reset_all(); 
    rng(prev_rng);

    assert (size(all_boxes, 1) == num_images);
    assert (size(all_boxes, 2) == num_classes);
    % ------------------------------------------------------------------------
    % Peform Corloc evaluation
    % ------------------------------------------------------------------------
    tic;
    [res] = corloc(num_classes, gt_boxes, all_boxes, 0.5);
    fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
    fprintf('Results:\n');
    res = res * 100;
    assert( numel(classes) == numel(res));
    for idx = 1:numel(res)
      fprintf('%12s : corloc : %5.2f\n', classes{idx}, res(idx));
    end
    fprintf('\nmean corloc : %.4f\n', mean(res));
    fprintf('~~~~~~~~~~~~~~~~~~~~ evaluate cost %.2f s\n', toc);
    mean_loc = mean(res);

    diary off;
end


% ------------------------------------------------------------------------
function [res] = corloc(num_class, gt_boxes, all_boxes, corlocThreshold)
% ------------------------------------------------------------------------
    num_image = numel(gt_boxes);     
    assert (num_image == size(all_boxes, 1));
    assert (num_class == size(all_boxes, 2));
    res = zeros(num_class, 1);
    for cls = 1:num_class
        overlaps = [];
        for idx = 1:num_image
           gt = gt_boxes{idx};
           gtboxes = gt.boxes(gt.class == cls, :);
           if (isempty(gtboxes)), continue; end
           localizedBox = all_boxes{idx,cls};
           overlap = iou(gtboxes, localizedBox);
           overlap = max(overlap);
           if (overlap >= corlocThreshold)
             overlaps(end+1) = 1;
           else
             overlaps(end+1) = 0;
           end
        end
        res(cls) = mean(overlaps);
    end
end
