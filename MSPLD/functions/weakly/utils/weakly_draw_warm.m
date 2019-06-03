function weakly_draw_warm(conf, roidb_train, cache_name)
  tic;
  save_dir = fullfile(conf.debug_cache_dir, cache_name);
  mkdir_if_missing(save_dir);
  classes = conf.classes;
  num_cls = length(classes);
  for index = 1:numel(roidb_train)
    split_name = roidb_train(index).image_id;
    if (isempty(strfind(split_name, 'flip')) == false), continue;end
    save_path = fullfile(save_dir, [split_name, '.jpg']);
    if (exist(save_path)), continue; end

    boxes_cell = cell(num_cls, 1);
    for i = 1:num_cls , boxes_cell{i} = zeros(0,5); end
    if (isempty(roidb_train(index).Debug_GT_Cls))
      %% Fully Sampled
      gt_index = find(roidb_train(index).GT_Index);
      cls      = roidb_train(index).class(gt_index,:);
      box      = roidb_train(index).boxes(gt_index,:);
    else 
      %% Unlabel data
      cls      = roidb_train(index).Debug_GT_Cls;
      box      = roidb_train(index).Debug_GT_Box;
    end

    for i = 1:length(cls)
      boxes_cell{cls(i)} = [boxes_cell{cls(i)}; [box(i,:), 1]];
    end

    %figure(1);
    im = imread(roidb_train(index).image_path);
    showboxes(im, boxes_cell, classes, 'voc');
    saveas(gcf, save_path);
  end
  fprintf('Finish Draw Debug Information : %s in %.1fs\n', cache_name, toc);
end
