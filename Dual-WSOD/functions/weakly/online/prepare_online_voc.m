function res_name = prepare_online_voc(cls, boxes, imdb, cache_name, suffix)
% res = imdb_eval_voc(cls, boxes, imdb, suffix)
%   Use the VOCdevkit to evaluate detections specified in boxes
%   for class cls against the ground-truth boxes in the image
%   database imdb. Results files are saved with an optional
%   suffix.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Add a random string ("salt") to the end of the results file name
% to prevent concurrent evaluations from clobbering each other
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

cache_dir = fullfile('output', 'weakly_cachedir', cache_name, [imdb.name, '_online_results']);
fprintf('prepare_online_voc cache_dir : %s\n', cache_dir);

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
copyfile(res_fn, cache_dir);
tempS = regexp(res_fn, '/', 'split');
res_name = fullfile(cache_dir, tempS{end});
fclose(fid);
%delete(res_fn);

rmpath(fullfile(VOCopts.datadir, 'VOCcode')); 
end
