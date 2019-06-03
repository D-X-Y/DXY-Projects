function roidb = roidb_for_test(imdb, varargin)
% --- roidb.test_boxes ------------------------------------

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('exclude_difficult_samples',       true,   @islogical);
ip.addParamValue('with_selective_search',           false,  @islogical);
ip.addParamValue('with_edge_box',                   false,  @islogical);
ip.addParamValue('with_self_proposal',              false,  @islogical);
ip.addParamValue('rootDir',                         '.',    @ischar);
ip.addParamValue('extension',                       '',     @ischar);
ip.parse(imdb, varargin{:});
opts = ip.Results;

roidb.name = imdb.name;
if ~isempty(opts.extension)
    opts.extension = ['_', opts.extension];
end
regions_file_ss = fullfile(opts.rootDir, sprintf('/data/selective_search_data/%s%s.mat', roidb.name, opts.extension));
regions_file_eb = fullfile(opts.rootDir, sprintf('/data/edge_box_data/%s%s.mat', roidb.name, opts.extension));
regions_file_sp = fullfile(opts.rootDir, sprintf('/data/self_proposal_data/%s%s.mat', roidb.name, opts.extension));

cache_file_ss = [];
cache_file_eb = [];
cache_file_sp = [];
if opts.with_selective_search 
    cache_file_ss = 'ss_';
    if~exist(regions_file_ss, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_ss);
    end
end

if opts.with_edge_box 
    cache_file_eb = 'eb_';
    if ~exist(regions_file_eb, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_eb);
    end
end

if opts.with_self_proposal 
    cache_file_sp = 'sp_';
    if ~exist(regions_file_sp, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_sp);
    end
end

cache_file = fullfile(opts.rootDir, ['/imdb/cache/roidb_for_test_' cache_file_ss cache_file_eb cache_file_sp imdb.name opts.extension]);
assert (imdb.flip == false);
cache_file = [cache_file, '.mat'];
try
  load(cache_file);
catch
  VOCopts = imdb.details.VOCopts;

  addpath(fullfile(VOCopts.datadir, 'VOCcode')); 

  roidb.name = imdb.name;

  fprintf('Loading region proposals...');
  regions = [];
  if opts.with_selective_search
        regions = load_proposals(regions_file_ss, regions);
  end
  if opts.with_edge_box
        regions = load_proposals(regions_file_eb, regions);
  end
  if opts.with_self_proposal
        regions = load_proposals(regions_file_sp, regions);
  end
  fprintf('done\n');
  if isempty(regions)
      fprintf('Warrning: no windows proposal is loaded !\n');
      regions.boxes = cell(length(imdb.image_ids), 1);
      regions.images = imdb.image_ids;
  end

  for i = 1:length(imdb.image_ids)
        tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
        try
          voc_rec = PASreadrecord(sprintf(VOCopts.annopath, imdb.image_ids{i}));
        catch
          voc_rec = [];
        end

        [~, image_name1] = fileparts(imdb.image_ids{i});
        [~, image_name2] = fileparts(regions.images{i});
        assert(strcmp(image_name1, image_name2));
        im = imread(imdb.image_at(i));
        imgsize = size(im);
        roidb.test_boxes{i} = obtain_proposals(imgsize(2:-1:1), regions.boxes{i});
  end

  rmpath(fullfile(VOCopts.datadir, 'VOCcode')); 

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end

end %%% For function


% ------------------------------------------------------------------------
function rec = obtain_proposals(imgsize, boxes) 
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
if ~isempty(boxes)
    boxes = boxes(:, [2 1 4 3]);
    assert (all(boxes(:,3) <= imgsize(1)));
    assert (all(boxes(:,4) <= imgsize(2)));
end
rec = boxes;
end

% ------------------------------------------------------------------------
function regions = load_proposals(proposal_file, regions)
% ------------------------------------------------------------------------
if isempty(regions)
    regions = load(proposal_file);
else
    regions_more = load(proposal_file);
    if ~all(cellfun(@(x, y) strcmp(x, y), regions.images(:), regions_more.images(:), 'UniformOutput', true))
        error('roidb_from_ilsvrc: %s is has different images list with other proposals.\n', proposal_file);
    end
    regions.boxes = cellfun(@(x, y) [double(x); double(y)], regions.boxes(:), regions_more.boxes(:), 'UniformOutput', false);
end
end
