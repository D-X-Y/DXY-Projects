clear; clc;
caffe.set_mode_gpu();
gpu_id = 5;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
net_model = '../models/bvlc_resnet101/ResNet-101-deploy.prototxt';
net_weights = '../models/bvlc_resnet101/ResNet-101-model.caffemodel';
net = caffe.Net(net_model, net_weights, 'test');
path = '/home/dongxuanyi/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt';
image_path = '/home/dongxuanyi/data/VOCdevkit2007/VOC2007/JPEGImages/';
[imagelist] = textread(path, '%s');
im_mean = load('../models/mean_image.mat');
im_mean = im_mean.image_mean;
target_size = [224, 224];

for index = 1:numel(imagelist)
    tic;
    image = fullfile(image_path, [imagelist{index}, '.jpg']);
    im = single(imread(image));
    im_means = imresize(im_mean, [size(im, 1), size(im, 2)], 'bilinear', 'antialiasing', false);
    im = bsxfun(@minus, im, im_means);
    im = imresize(im, target_size, 'bilinear', 'antialiasing', false);
    im_blob = im(:, :, [3, 2, 1]); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3]));
    scores{index} = net.forward({im_blob});
    res5c{index} = net.blobs('pool5').get_data();
    fprintf('Handle %d / %d , cost %.3f\n', index, numel(imagelist), toc);
end
caffe.reset_all();
save('./VOC07_Test_res101.mat', 'res5c', 'imagelist', '-v7.3');
