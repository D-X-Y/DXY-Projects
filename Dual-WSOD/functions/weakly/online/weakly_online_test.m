function res = weakly_online_test(confs, imdb, roidb, varargin)
% --------------------------------------------------------
% Dual-Network implementation
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
    cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, [imdb.name, '_online_results']);
    mkdir_if_missing(cache_dir);
    fprintf('weakly_online_test cache_dir : %s\n', cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file  = fullfile(cache_dir, 'log', [opts.log_prefix, 'test_', timestamp, '.txt']);
    save_file = fullfile(cache_dir, 'log', [opts.log_prefix, 'test_', timestamp, '.mat']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    num_classes = imdb.num_classes;
    caffe.reset_all(); 
    
%%  testing 
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

    num_nets   = numel(opts.net_defs);

    aboxes_map = cell(num_classes);
    for i = 1:num_classes
        aboxes_map{i} = cell(num_images, 1);
    end

    t_start = tic;


    for inet = 1:num_nets
      caffe.reset_all();
      caffe_net = caffe.Net(opts.net_defs{inet}, 'test');
      caffe_net.copy_from(opts.net_models{inet});
      fprintf('>>> %3d / %3d net-prototxt:: %s\n', inet, num_nets, opts.net_defs{inet});
      fprintf('>>> %3d / %3d net-weights :: %s\n', inet, num_nets, opts.net_models{inet});

      for i = 1:num_images

        if (rem(i, opts.dis_itertion) == 1), fprintf('%s: test (%s) %d/%d cost : %.1f s\n', procid(), imdb.name, i, num_images, toc(t_start)); end
        im = imread(imdb.image_at(i));
        pre_boxes = roidb.test_boxes{i};
        [boxes, scores] = weakly_im_detect(confs{inet}, caffe_net, im, pre_boxes, confs{inet}.max_rois_num_in_gpu);

        for j = 1:num_classes
            cls_boxes = boxes(:, (1+(j-1)*4):((j)*4));
            cls_scores = scores(:, j);
            temp = cat(2, single(cls_boxes), single(cls_scores));

            if (isempty(aboxes_map{j}{i}))
                aboxes_map{j}{i} = temp ./ num_nets;
            else
                aboxes_map{j}{i} = aboxes_map{j}{i} + temp ./ num_nets;
            end
        end
      end
    end
    caffe.reset_all(); 
    rng(prev_rng);

    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------
    %%% For mAP
    assert (isequal(imdb.eval_func, @imdb_eval_voc));
    tic;

    %  only for pascal voc
    res = cell(num_classes, 1);
    parfor model_ind = 1:num_classes
        cls = imdb.classes{model_ind};
        res{model_ind} = prepare_online_voc(cls, aboxes_map{model_ind}, imdb, opts.cache_name, opts.suffix);
    end

    
    diary off;
end


% ------------------------------------------------------------------------
function [boxes] = keep_top_k(boxes, top_k)
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
