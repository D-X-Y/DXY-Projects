function dataset = voc2007_trainval_ss(dataset, usage, use_flip, exclude_difficult_samples)
% Pascal voc 0712 trainval set with selective search
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit2007                      = voc2007_devkit();

switch usage
    case {'train'}
        dataset.imdb_train    = { imdb_from_voc(devkit2007, 'trainval', '2007', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, 'with_selective_search', true, 'exclude_difficult_samples', exclude_difficult_samples), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        error('only supports one source test currently');  
    otherwise
        error('usage = ''train'' or ''test''');
end

end
