clear;clc;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(7);

cifar10_train_data = load(fullfile('caffe','examples','cifar10','cifar10_train_lmdb.mat'));
cifar10_train_data.image = single(permute(cifar10_train_data.image,[3,2,4,1]));
cifar10_test_data = load(fullfile('caffe','examples','cifar10','cifar10_test_lmdb.mat'));
cifar10_test_data.image = single(permute(cifar10_test_data.image,[3,2,4,1]));
train_num = size(cifar10_train_data.label, 1);
test_num = size(cifar10_test_data.label, 1);

mean_cifar10 = load('./caffe/models/mean_cifar10.mat');
mean_cifar10 = mean_cifar10.mean;
mean_cifar10 = permute(mean_cifar10,[3,2,1]);

%sub mean
cifar10_train_data.image = cifar10_train_data.image - single(repmat(mean_cifar10,1,1,1,train_num));
cifar10_test_data.image = cifar10_test_data.image - single(repmat(mean_cifar10,1,1,1,test_num));

assert(train_num == 50000);
assert(test_num == 10000);
cifar10_train_data.label = reshape(cifar10_train_data.label, [1,1,1,numel(cifar10_train_data.label)]);
cifar10_test_data.label = reshape(cifar10_test_data.label, [1,1,1,numel(cifar10_test_data.label)]);

net = [];
blob_names = [];
%%%% 
net_model = './caffe/models/cifar10_ori/cifar10_full.prototxt';
net_weights = './caffe/models/cifar10_ori/cifar10_full_iter_70000.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool3';

net_model = './caffe/models/cifar10_res20/deploy.prototxt';
net_weights = './caffe/models/cifar10_res20/cifar10_res20_iter_70000.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'global_pool';

net_model = './caffe/models/cifar10_res110/deploy.prototxt';
net_weights = './caffe/models/cifar10_res110/cifar10_res110_iter_68000.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'global_pool';

net_model = './caffe/models/nin/half.prototxt';
net_weights = './caffe/models/nin/cifar10_nin.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool3';

net_model = './caffe/models/cifar10_164/deploy.prototxt';
net_weights = './caffe/models/cifar10_164/cifar10_res164_iter_70000.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'Pooling1';

net_model = './caffe/models/bvlc_googlenet/half.prototxt';
net_weights = './caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool3/3x3_s2';

net_model = './caffe/models/bvlc_vgg16/half.prototxt';
net_weights = './caffe/models/bvlc_vgg16/VGG16.v2.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'pool5';

net_model = './caffe/models/cifar10_56/deploy.prototxt';
net_weights = './caffe/models/cifar10_56/cifar10_res56_iter_70000.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'Pooling1';

net_model = './caffe/models/cifar10_44/deploy.prototxt';
net_weights = './caffe/models/cifar10_44/cifar10_res44_iter_70000.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'global_pool';

net_model = './caffe/models/cifar10_32/deploy.prototxt';
net_weights = './caffe/models/cifar10_32/cifar10_res32_iter_70000.caffemodel';
net{end+1} = caffe.Net(net_model, net_weights, 'test');
blob_names{end+1} = 'global_pool';
train_features = cell(numel(net), train_num);
test_features = cell(numel(net), test_num);
tic;

fprintf('train : %d, test : %d, features : %d\n', train_num, test_num, numel(net));
for index = 1:train_num
    data = cifar10_train_data.image(:,:,:,index);
    label = cifar10_train_data.label(:,:,:,index);
    for jj = 1:numel(net)
        net{jj}.blobs('data').set_data(data);
        net{jj}.forward_prefilled();
        cur = net{jj}.blobs(blob_names{jj}).get_data();
        train_features{jj}{index} = reshape(cur, 1, numel(cur));
    end
    if (toc >= 10)
        fprintf('%04d / %04d train data convert done, cost %.2f s\n', index, train_num, toc);
        tic;
    end
end
tic;
for index = 1:test_num
    data = cifar10_test_data.image(:,:,:,index);
    label = cifar10_test_data.label(:,:,:,index);
    for jj = 1:numel(net)
        net{jj}.blobs('data').set_data(data);
        net{jj}.forward_prefilled();
        cur = net{jj}.blobs(blob_names{jj}).get_data();
        test_features{jj}{index} = reshape(cur, 1, numel(cur));
    end
    if (toc > 10)
        fprintf('%04d / %04d test data convert done, cost %.2f s\n', index, test_num, toc);
        tic;
    end
end

%%%%
caffe.reset_all();
clear index jj data label;
TTT = cell(numel(net), 1);
for i = 1:numel(net)
    AA = zeros(test_num, numel(test_features{i}{1}));
    for j = 1:test_num
        AA(j,:)  = test_features{i}{j};
    end
    TTT{i} = AA;
end
test_features = TTT;
TTT = cell(numel(net), 1);
for i = 1:numel(net)
    AA = zeros(train_num, numel(train_features{i}{1}));
    for j = 1:train_num
        AA(j,:)  = train_features{i}{j};
    end
    TTT{i} = AA;
end
train_features = TTT;
clear TTT;
train_label = reshape(cifar10_train_data.label, train_num, 1);
test_label = reshape(cifar10_test_data.label, test_num, 1);
