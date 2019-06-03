%{
path = '/home/dongxuanyi/AAAI/features/wipedia_articles/images/';
name = 'wipedia_articles';
path = '/home/dongxuanyi/AAAI/features/pascal_sentences/images/';
name = 'pascal_sentences';
%}
function dataset = GenerateData(name, path)
    clearvars -except name path;
    dataset.name = name;
    assert( exist(path, 'dir') == 7);
    dirs = dir(path);  
    dir_names = {dirs.name};
    dir_names = setdiff(dir_names, {'.', '..', '.DS_Store'});
    dataset.synset = dir_names;
    dataset.labels = numel(dir_names);
    image_list = [];
    image_label = [];
    for index = 1:numel(dir_names)
        images = dir(fullfile(path, dir_names{index}, '*.jpg'));
        for j = 1:numel(images)
            image_list{end+1} = fullfile(name, 'images', dir_names{index}, images(j).name);
            image_label(end+1) = index;
        end
    end
    dataset.image_list = image_list';
    dataset.image_label = image_label';
end
