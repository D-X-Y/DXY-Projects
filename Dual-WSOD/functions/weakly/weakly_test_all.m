function [mAP, mean_loc] = weakly_test_all(confs, imdb, roidb, varargin)
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
    cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, [imdb.name, '_mAP']);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file  = fullfile(cache_dir, 'log', [opts.log_prefix, 'test_', timestamp, '.txt']);
    save_file = fullfile(cache_dir, 'log', [opts.log_prefix, 'test_', timestamp, '.mat']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    num_classes = imdb.num_classes;
    caffe.reset_all(); 
    
    %%% Ground Truth for corloc
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

    aboxes_map = cell(num_classes, num_nets);
    for i = 1:num_classes
        for j = 1:num_nets
          aboxes_map{i, j} = cell(num_images, 1);
        end
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
        d = roidb.rois(i);
        im = imread(imdb.image_at(i));
        pre_boxes = d.boxes(~d.gt, :);
        [boxes, scores] = weakly_im_detect(confs{inet}, caffe_net, im, pre_boxes, confs{inet}.max_rois_num_in_gpu);

        for j = 1:num_classes
            cls_boxes = boxes(:, (1+(j-1)*4):((j)*4));
            cls_scores = scores(:, j);
            %temp = cat(2, single(cls_boxes), single(cls_scores));
            aboxes_map{j, inet}{i} = cat(2, single(cls_boxes), single(cls_scores));
        end
      end
    end
    caffe.reset_all(); 
    rng(prev_rng);

    % ------------------------------------------------------------------------
    % Peform Corloc evaluation
    % ------------------------------------------------------------------------
    tic;
    aboxes_cor = cell(num_images, 1);

    res_cor_net = cell(num_nets, 1);
    for inet = 1:num_nets

        %%% Calculate 
        aboxes_cor_net = cell(num_images, 1);
        for i = 1:num_images
          cor_boxes = zeros(0, 4); 
          for cls = 1:num_classes
            taboxes = aboxes_map{cls, inet}{i};
            tscore = taboxes(:, 5); 
            tboxes = taboxes(:, 1:4);
            [~, idx] = max(tscore);
            cor_boxes = [cor_boxes; tboxes(idx,:)];
          end 
          aboxes_cor_net{i} = cor_boxes;

          if (isempty(aboxes_cor{i}))
            aboxes_cor{i} = cor_boxes ./ num_nets;
          else
            aboxes_cor{i} = aboxes_cor{i} + cor_boxes ./ num_nets;
          end
        end 
        [res_cor_net{inet}] = corloc(num_classes, gt_boxes, aboxes_cor_net, 0.5);
        res_cor_net{inet} = res_cor_net{inet} * 100;
    end

    [res_cor] = corloc(num_classes, gt_boxes, aboxes_cor, 0.5);
    fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
    fprintf('Fusion CorLoc Results:\n');
    res_cor = res_cor * 100;
    assert( numel(classes) == numel(res_cor));
    for idx = 1:numel(res_cor)
      if (num_nets == 2)
        fprintf('%12s : corloc : %5.2f  [%5.2f , %5.2f ]\n', classes{idx}, res_cor(idx), res_cor_net{1}(idx), res_cor_net{2}(idx));
      else
        fprintf('%12s : corloc : %5.2f\n', classes{idx}, res_cor(idx));
      end
    end 
    if (num_nets == 2)
      fprintf('\nmean corloc : %.4f  [%.4f  , %.4f]\n', mean(res_cor), mean(res_cor_net{1}), mean(res_cor_net{2}));
    else
      fprintf('\nmean corloc : %.4f\n', mean(res_cor));
    end
    fprintf('~~~~~~~~~~~~~~~~~~~~ evaluate cost %.2f s\n', toc);
    mean_loc = mean(res_cor);



    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------
    %%% For mAP
    res_map_net = cell(num_nets, 1);
    assert (isequal(imdb.eval_func, @imdb_eval_voc));
    tic;

    aboxes_map_fusion = cell(num_classes, 1);
    for i = 1:num_classes
      aboxes_map_fusion{i} = cell(num_images, 1);
    end

    for inet = 1:num_nets
      aboxes_map_net = cell(num_classes, 1);
      for i = 1:num_classes
        aboxes_map_net{i} = cell(num_images, 1);
      end

      for j = 1:num_classes
        [aboxes_map_net{j}] = keep_top_k(aboxes_map{j, inet}, max_per_set);
      end
      temp = cell(num_classes, 1); 
      parfor model_ind = 1:num_classes
          cls = imdb.classes{model_ind};
          temp{model_ind} = imdb.eval_func(cls, aboxes_map_net{model_ind}, imdb, opts.cache_name, opts.suffix);
      end
      temp = cat(1, temp{:});   temp = [temp(:).ap]' * 100;
      res_map_net{inet} = temp;
  
      %% Fusion
      for j = 1:num_classes
        for i = 1:num_images
          if (isempty(aboxes_map_fusion{j}{i}))
            aboxes_map_fusion{j}{i} = aboxes_map{j, inet}{i} ./ num_nets;
          else
            aboxes_map_fusion{j}{i} = aboxes_map{j, inet}{i} ./ num_nets + aboxes_map_fusion{j}{i};
          end
        end
      end
      %% END Fusion
    end
    aboxes_map = cell(num_classes, 1);
    for j = 1:num_classes
        [aboxes_map{j}] = keep_top_k(aboxes_map_fusion{j}, max_per_set);
    end

    res_map = cell(num_classes, 1);
    parfor model_ind = 1:num_classes
        cls = imdb.classes{model_ind};
        res_map{model_ind} = imdb.eval_func(cls, aboxes_map{model_ind}, imdb, opts.cache_name, opts.suffix);
    end
    res_map = cat(1, res_map{:});
    


    if ~isempty(res_map)
        fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
        fprintf('Fusion meanAP Results:\n');
        aps = [res_map(:).ap]' * 100;
        %disp(aps);
        assert( numel(classes) == numel(aps));
        for idx = 1:numel(aps)
            if (num_nets == 2)
              fprintf('%12s : %5.2f  [%5.2f , %5.2f]\n', classes{idx}, aps(idx), res_map_net{1}(idx), res_map_net{2}(idx));
            else
              fprintf('%12s : %5.2f\n', classes{idx}, aps(idx));
            end
        end
        if (num_nets == 2)
          fprintf('\nmean mAP : %.4f  [%.4f , %.4f]\n', mean(aps), mean(res_map_net{1}), mean(res_map_net{2}));
        else
          fprintf('\nmean mAP : %.4f\n', mean(aps));
        end
        %disp(mean(aps));
        fprintf('~~~~~~~~~~~~~~~~~~~~ evaluate cost %.2f s\n', toc);
        mAP = mean(aps);
    else
        mAP = nan;
    end

    save(save_file, 'res_cor_net', 'res_cor', 'mean_loc', 'res_map_net', 'res_map', 'mAP', '-v7.3');

    %%% Print For Latex
    fprintf('CorLoc :  ');
    for j = 1:numel(res_cor)
      fprintf('%.1f & ', res_cor(j));
    end
    fprintf('   %.1f \nmeanAP :  ', mean_loc);
    for j = 1:numel(aps)
      fprintf('%.1f & ', aps(j));
    end
    fprintf('   %.1f \n', mAP);

    
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

% ------------------------------------------------------------------------
function [res] = corloc(num_class, gt_boxes, all_boxes, corlocThreshold)
% ------------------------------------------------------------------------
    num_image = numel(gt_boxes);     assert (num_image == numel(all_boxes));
    res = zeros(num_class, 1);
    for cls = 1:num_class
        overlaps = [];
        for idx = 1:num_image
           gt = gt_boxes{idx};
           gtboxes = gt.boxes(gt.class == cls, :);
           if (isempty(gtboxes)), continue; end
           localizedBox = all_boxes{idx}(cls, :);
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
