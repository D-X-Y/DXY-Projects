function dataset = noise_voc07_ss(dataset, num_noise, use_flip)
  devkit      = noise_devkit();
  devkit2007  = voc2007_devkit();
  dataset.imdb_train = { imdb_from_voc(devkit2007, 'trainval', '2007', use_flip), ...
                          imdb_from_noise(devkit, num_noise, use_flip) };
  dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, 'with_selective_search', true), dataset.imdb_train, 'UniformOutput', false);
end
