% load('/home/dongxuanyi/AAAI/features/pascal_sentences/pascal_sentences.mat');
% load('/home/dongxuanyi/AAAI/features/wipedia_articles/wipedia_articles.mat');
function [features, label] = GenerateFeatures(dataset, gpu_id)
caffe_path = 'caffe/matlab';
addpath (caffe_path);
path = '/home/dongxuanyi/AAAI/features/';
%Temp = load(fullfile(path, 'pascal_sentences/pascal_sentences.mat'));
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
image_list = dataset.image_list;
label = dataset.image_label;
assert(numel(image_list) == numel(label));
total = numel(image_list);
% caffe net
im_mean = load('./caffe/models/mean_image.mat');
im_mean = im_mean.image_mean;
net = [];
blob_names = [];
target_sizes = [];
net_model = './caffe/models/bvlc_vgg16/VGG_ILSVRC_16_layers_deploy.txt';
net_weights = './caffe/models/bvlc_vgg16/VGG16.v2.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'fc7';
target_sizes{end+1} = [224,224];
net_model = './caffe/models/bvlc_googlenet/deploy.prototxt';
net_weights = './caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool5/7x7_s1';
target_sizes{end+1} = [224,224];

net_model = './caffe/models/bvlc_resnet101/ResNet-101-deploy.prototxt';
net_weights = './caffe/models/bvlc_resnet101/ResNet-101-model.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool5';
target_sizes{end+1} = [224,224];
net_model = './caffe/models/bvlc_resnet152/ResNet-152-deploy.prototxt';
net_weights = './caffe/models/bvlc_resnet152/ResNet-152-model.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool5';
target_sizes{end+1} = [224,224];
net_model = './caffe/models/bvlc_alexnet/deploy.prototxt';
net_weights = './caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'fc6';
target_sizes{end+1} = [227,227];
net_model = './caffe/models/bvlc_reference_caffenet/deploy.prototxt';
net_weights = './caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool5';
target_sizes{end+1} = [227,227];

features = cell(numel(net), total);
for index = 1:total
    tic;
    image_path = fullfile(path, image_list{index});
    im         = imread(image_path);
    %hog{index} = HOG(im, target_size);
    for c_net = 1:numel(net)
        features{c_net}{index} = CAFFE(net{c_net}, im_mean, im, target_sizes{c_net}, blob_names{c_net});
    end
    fprintf('%04d / %04d, cost %.3f s \n', index, total, toc);
end

%[train_id, test_id] = split_data(total, 0.8);
%model = train(label(train_id), sparse(features(train_id,:)));
%[predicted_label, accuracy, prob_estimates] = predict(label(test_id), sparse(features(test_id,:)), model);

caffe.reset_all();
rmpath (caffe_path);
end

function hog = HOG(im, target_size)
    im = imresize(im, target_size, 'bilinear', 'antialiasing', false);
    cellSize = 8 ;
    hog = vl_hog(im, cellSize) ;
end

function sift = SIFT(im, target_size)
    im = single(rgb2gray(im));
    im = imresize(im, target_size, 'bilinear', 'antialiasing', false);
    [d, sift] = vl_sift(im);
end

function feature = CAFFE(net, im_mean, im, target_size, blob_name)
    im = single(im);
    im_means = imresize(im_mean, [size(im, 1), size(im, 2)], 'bilinear', 'antialiasing', false);
    im = bsxfun(@minus, im, im_means);
    im = imresize(im, target_size, 'bilinear', 'antialiasing', false);
    im_blob = im(:, :, [3, 2, 1]); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3]));
    [~] = net.forward({im_blob});
    feature = net.blobs(blob_name).get_data();
    feature = reshape(feature, 1, numel(feature));
end
