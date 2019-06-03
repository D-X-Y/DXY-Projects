function startup()
% startup()
% --------------------------------------------------------
% MSPLD implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2018, Xuanyi Dong
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    curdir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(curdir, 'utils')));
    addpath(genpath(fullfile(curdir, 'functions')));
    addpath(genpath(fullfile(curdir, 'bin')));
    addpath(genpath(fullfile(curdir, 'experiments')));
    addpath(genpath(fullfile(curdir, 'imdb')));
    addpath(genpath(fullfile(curdir, 'data')));

    mkdir_if_missing(fullfile(curdir, 'datasets'));

    mkdir_if_missing(fullfile(curdir, 'external'));

    caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab');
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));

    mkdir_if_missing(fullfile(curdir, 'imdb', 'cache'));

    mkdir_if_missing(fullfile(curdir, 'output'));

    mkdir_if_missing(fullfile(curdir, 'models'));

    mspld_build();
    addpath(fullfile(curdir, 'selective_search'));
    if exist('selective_search/SelectiveSearchCodeIJCV')
        addpath(fullfile(curdir, 'selective_search/SelectiveSearchCodeIJCV'));
        addpath(fullfile(curdir, 'selective_search/SelectiveSearchCodeIJCV/Dependencies'));
        ss_build();
        fprintf('Compile SelectiveSearchCodeIJCV done\n');
    else
        fprintf('Warning: you will need the selective search IJCV code.\n');
    end

    fprintf('MSPLD startup done\n');
end
