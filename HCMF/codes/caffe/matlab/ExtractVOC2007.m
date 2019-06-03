caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
net_model = '../models/bvlc_vgg16/VGG_ILSVRC_16_layers_deploy.txt';
net_weights = '../models/bvlc_vgg16/VGG16.v2.caffemodel';
net = caffe.Net(net_model, net_weights, 'test');
path = '/home/dongxuanyi/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt';
save_path = './VOC07_Trainval.mat';
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
    scores = net.forward({im_blob});
    fc7{index} = net.blobs('fc7').get_data();
    fprintf('Handle %d / %d , cost %.3f\n', index, numel(imagelist), toc);
end
caffe.reset_all();
save(save_path, 'fc7', 'imagelist', '-v7.3');
