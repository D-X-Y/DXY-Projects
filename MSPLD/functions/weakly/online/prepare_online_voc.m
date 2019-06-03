function res_name = prepare_online_voc(cls, boxes, imdb, cache_name, suffix)
% --------------------------------------------------------
% MSPLD implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

use_res_salt = true;
% comp4 because we use outside data (ILSVRC2012)
comp_id = 'comp4';
% draw each class curve
draw_curve = true;

% save results
if ~exist('suffix', 'var') || isempty(suffix) || strcmp(suffix, '')
  suffix = '';
else
  if suffix(1) ~= '_'
    suffix = ['_' suffix];
  end
end

conf.cache_dir = fullfile('output', 'weakly_cachedir', cache_name, [imdb.name, '_online_results']);

VOCopts  = imdb.details.VOCopts;
image_ids = imdb.image_ids;
test_set = VOCopts.testset;
year = VOCopts.dataset(4:end);

addpath(fullfile(VOCopts.datadir, 'VOCcode')); 

if use_res_salt
  prev_rng = rng;
  rng shuffle;
  salt = sprintf('%d', randi(100000));
  res_id = [comp_id '-' salt];
  rng(prev_rng);
else
  res_id = comp_id;
end
res_fn = sprintf(VOCopts.detrespath, res_id, cls);

% write out detections in PASCAL format and score
fid = fopen(res_fn, 'w');
for i = 1:length(image_ids)
  bbox = boxes{i};
  keep = nms(bbox, 0.3);
  bbox = bbox(keep,:);
  for j = 1:size(bbox,1)
    fprintf(fid, '%s %f %.3f %.3f %.3f %.3f\n', image_ids{i}, bbox(j,end), bbox(j,1:4));
  end
end
copyfile(res_fn, conf.cache_dir);
tempS = regexp(res_fn, '/', 'split');
res_name = fullfile(conf.cache_dir, tempS{end});
fclose(fid);
delete(res_fn);

rmpath(fullfile(VOCopts.datadir, 'VOCcode')); 
end
