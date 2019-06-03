clear;clc;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(3);

cifar100_train_data = load(fullfile('caffe','examples','cifar100','cifar100_train_lmdb.mat'));
cifar100_train_data.image = single(permute(cifar100_train_data.image,[3,2,4,1]));
cifar100_test_data = load(fullfile('caffe','examples','cifar100','cifar100_test_lmdb.mat'));
cifar100_test_data.image = single(permute(cifar100_test_data.image,[3,2,4,1]));
train_num = size(cifar100_train_data.label, 1);
test_num = size(cifar100_test_data.label, 1);

mean_cifar100 = load('./caffe/models/mean_cifar100.mat');
mean_cifar100 = mean_cifar100.mean;
mean_cifar100 = permute(mean_cifar100,[3,2,1]);

%sub mean
cifar100_train_data.image = cifar100_train_data.image - single(repmat(mean_cifar100,1,1,1,train_num));
cifar100_test_data.image = cifar100_test_data.image - single(repmat(mean_cifar100,1,1,1,test_num));

assert(train_num == 50000);
assert(test_num == 10000);
cifar100_train_data.label = reshape(cifar100_train_data.label, [1,1,1,numel(cifar100_train_data.label)]);
cifar100_test_data.label = reshape(cifar100_test_data.label, [1,1,1,numel(cifar100_test_data.label)]);

net        = [];
blob_names = [];
net_model  = [];
net_weights= [];
%%%% 
net_model{end+1} = './caffe/models/PO_32/cifar100_res32_trainval.proto';
net_weights{end+1} = './caffe/models/PO_32/cifar100_res32_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'global_pool';
net_model{end+1} = './caffe/models/PO_44/cifar100_res44_trainval.proto';
net_weights{end+1} = './caffe/models/PO_44/cifar100_res44_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'global_pool';
net_model{end+1} = './caffe/models/PO_56/cifar100_res56_trainval.proto';
net_weights{end+1} = './caffe/models/PO_56/cifar100_res56_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'global_pool';
net_model{end+1} = './caffe/models/PO_110/cifar100_res110_trainval.proto';
net_weights{end+1} = './caffe/models/PO_110/cifar100_res110_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'global_pool';
net_model{end+1} = './caffe/models/PI_20/cifar100_res20_trainval.proto';
net_weights{end+1} = './caffe/models/PI_20/cifar100_res20_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'Pooling1';
net_model{end+1} = './caffe/models/PI_32/cifar100_res32_trainval.proto';
net_weights{end+1} = './caffe/models/PI_32/cifar100_res32_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'Pooling1';
net_model{end+1} = './caffe/models/PI_44/cifar100_res44_trainval.proto';
net_weights{end+1} = './caffe/models/PI_44/cifar100_res44_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'Pooling1';
net_model{end+1} = './caffe/models/PI_56/cifar100_res56_trainval.proto';
net_weights{end+1} = './caffe/models/PI_56/cifar100_res56_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'Pooling1';
net_model{end+1} = './caffe/models/PI_110/cifar100_res110_trainval.proto';
net_weights{end+1} = './caffe/models/PI_110/cifar100_res110_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'Pooling1';
net_model{end+1} = './caffe/models/PI_164/cifar100_res164_trainval.proto';
net_weights{end+1} = './caffe/models/PI_164/cifar100_res164_iter_64000.caffemodel';
net{end+1} = caffe.Net(net_model{end}, net_weights{end}, 'test');
blob_names{end+1} = 'Pooling1';
tic;
train_features = cell(numel(net), train_num);
test_features = cell(numel(net), test_num);

fprintf('train : %d, test : %d, features : %d\n', train_num, test_num, numel(net));
for index = 1:train_num
    data = cifar100_train_data.image(:,:,:,index);
    label = cifar100_train_data.label(:,:,:,index);
    for jj = 1:numel(net)
        net{jj}.blobs('data').set_data(data);
        net{jj}.forward_prefilled();
        cur = net{jj}.blobs(blob_names{jj}).get_data();
        train_features{jj}{index} = reshape(cur, 1, numel(cur));
    end
    if (toc >= 60)
        fprintf('%04d / %04d train data convert done, cost %.2f s\n', index, train_num, toc);
        tic;
    end
end
tic;
for index = 1:test_num
    data = cifar100_test_data.image(:,:,:,index);
    label = cifar100_test_data.label(:,:,:,index);
    for jj = 1:numel(net)
        net{jj}.blobs('data').set_data(data);
        net{jj}.forward_prefilled();
        cur = net{jj}.blobs(blob_names{jj}).get_data();
        test_features{jj}{index} = reshape(cur, 1, numel(cur));
    end
    if (toc > 60)
        fprintf('%04d / %04d test data convert done, cost %.2f s\n', index, test_num, toc);
        tic;
    end
end

BACK_TRAIN = train_features;
BACK_TEST  = test_features;
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
train_label = reshape(cifar100_train_data.label, train_num, 1);
test_label = reshape(cifar100_test_data.label, test_num, 1);
