function voc_2012 = prepare_submit_voc(res_path, cache_name, dir_name, thresh)

  cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', cache_name, dir_name);
  mkdir_if_missing(cache_dir);
  voc_2012  = fullfile(cache_dir, 'VOC2012');

  main_dir  = fullfile(voc_2012, 'Main');

  mkdir_if_missing(main_dir);
  for i = 1:numel(res_path)
    cur_path = res_path{i};
    temp = regexp(cur_path, 'comp4-', 'split');
    temp = regexp(temp{2}, '_', 'split');
    to_path  = fullfile(main_dir, ['comp4_', 'det_test_', temp{end}]);
    %copyfile(cur_path, to_path);
    [image_ids, score, x1, y1, x2, y2] = textread(cur_path, '%s %f %f %f %f %f');
    bbox = [x1, y1, x2, y2];
    ids = find(score > thresh);
    image_ids = image_ids(ids);
    bbox = bbox(ids, :);
    score = score(ids);

    fid = fopen(to_path, 'w');
    for j = 1:size(bbox,1)
      fprintf(fid, '%s %f %.3f %.3f %.3f %.3f\n', image_ids{j}, score(j), bbox(j,1:4));
    end
    fclose(fid);
  end
  
end
