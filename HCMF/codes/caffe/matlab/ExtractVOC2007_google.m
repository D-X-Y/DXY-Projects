clear; clc;
caffe.set_mode_gpu();
gpu_id = 3;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
net_model = '../models/bvlc_googlenet/deploy.prototxt';
net_weights = '../models/bvlc_googlenet/bvlc_googlenet.caffemodel';
net = caffe.Net(net_model, net_weights, 'test');
path = '/home/dongxuanyi/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt';
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
    A = net.blobs('pool5/7x7_s1').get_data();
    inception_5b_o{index} = reshape(A, 1, numel(A));
    fprintf('Handle %d / %d , cost %.3f\n', index, numel(imagelist), toc);
end
caffe.reset_all();
save('./VOC07_Train_Googlenet.mat', 'inception_5b_o', 'imagelist', '-v7.3');
