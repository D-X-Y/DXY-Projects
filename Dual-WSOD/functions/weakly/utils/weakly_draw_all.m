function weakly_draw_all(image_roidb_train, cache_name, dir_name, classes)
  cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', cache_name, dir_name);
  mkdir_if_missing(cache_dir);
  fprintf('weakly_draw_all cache_dir : %s\n', cache_dir);
  for i = 1:numel(image_roidb_train)
    weakly_debug_final(classes, cache_dir, image_roidb_train(i));
  end

end
