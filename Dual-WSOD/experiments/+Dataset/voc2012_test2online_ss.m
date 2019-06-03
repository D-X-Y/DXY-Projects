function dataset = voc2012_test2online_ss(dataset, usage, use_flip)
% Pascal voc 2012 test set with selective search
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit                      = voc2012_devkit();

switch usage
    case {'test'}
        dataset.imdb_test     = imdb_from_voc(devkit, 'test', '2012', use_flip) ;
        dataset.roidb_test    = roidb_for_test(dataset.imdb_test, 'with_selective_search', true);
    otherwise
        error('usage = ''train'' or ''test''');
end

end
