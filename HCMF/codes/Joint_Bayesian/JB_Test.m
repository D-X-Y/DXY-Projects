load ('Test_ALL.mat');
train_labels = [train_label + 1; train_label + 1];             clear train_label;
X            = double([train_features{1}; train_features{2}]); clear train_features;
X            = sqrt(X);
train_x      = bsxfun(@rdivide,X,sum(X,2));
train_mean   = mean(train_x, 1);
%[COEFF,SCORE] = princomp(X,'econ');
%[COEFF, SCORE] = pca(X,'Algorithm','svd','Economy', true);
%fprintf('PCA done');

[mappedX, mapping] = JointBayesian(train_x, train_labels);
fprintf('JointBayesian Done\n');

% Dis_matrix = repmat(mapping.c,1,size(train_x,1))+repmat(mapping.c,1,size(train_x,1))+train_x * mapping.G *train_x';
[classes, bar, labels] = unique(train_labels);
nc                     = length(classes);
train_Intra            = zeros(nc*2, 2);

for i=1:nc
    train_Intra(2*i-1,:) = randperm(sum(train_labels == i),2) + find(train_labels == i,1,'first') - 1;
    train_Intra(2*i,:)   = randperm(sum(train_labels == i),2) + find(train_labels == i,1,'first') - 1;
end;
train_Extra = reshape(randperm(length(train_labels), 20000), 10000, 2);
train_Extra(train_labels(train_Extra(:,1)) == train_labels(train_Extra(:,2)),:) = [];
train_Extra(size(train_Intra,1)+1:end,:) = [];
Dis_train_Intra = zeros(size(train_Intra,1),1);
Dis_train_Extra = zeros(size(train_Intra,1),1);

for i=1:size(train_Intra,1)
    Dis_train_Intra(i) = train_x(train_Intra(i,1),:) * mapping.A * train_x(train_Intra(i,1),:)' + train_x(train_Intra(i,2),:) * mapping.A * train_x(train_Intra(i,2),:)' - 2 * train_x(train_Intra(i,1),:) * mapping.G * train_x(train_Intra(i,2),:)';
    Dis_train_Extra(i) = train_x(train_Extra(i,1),:) * mapping.A * train_x(train_Extra(i,1),:)' + train_x(train_Extra(i,2),:) * mapping.A * train_x(train_Extra(i,2),:)' - 2 * train_x(train_Extra(i,1),:) * mapping.G * train_x(train_Extra(i,2),:)';
end;
group_train = [ones(size(Dis_train_Intra,1),1);zeros(size(Dis_train_Extra,1),1)];
training = [Dis_train_Intra;Dis_train_Extra];

test_labels  = [test_label + 1;  test_label + 1];             clear test_label;
test_x       = double([test_features{1};  test_features{2}]); clear test_features;
test_x = double(test_x);
test_x = sqrt(normX);
test_x = bsxfun(@rdivide, normX, sum(normX,2));
test_x = bsxfun(@minus, normX, train_mean);
%normX = normX * COEFF(:,1:top_dims);

%test_Intra = pairlist_lfw.IntraPersonPair;
%test_Extra = pairlist_lfw.ExtraPersonPair;

test_Intra = zeros(5000, 2);
test_Extra = zeros(5000, 2);
for i = 1:5000
  
end


index_1_ok = train_labels == 1;
index_1_no = train_labels ~= 1;

result_Intra = zeros(5000,1);
result_Extra = zeros(5000,1);
for i=1:5000
  result_Intra(i) = normX(test_Intra(i,1),:) * mapping.A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * mapping.A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
  result_Extra(i) = normX(test_Extra(i,1),:) * mapping.A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * mapping.A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
end;

group_sample = [ones(5000,1); zeros(5000,1)];
sample = [result_Intra;result_Extra];

%method SVM
bestc=256;
% bestg=128;
% cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg)];
%cmd = [' -t 0 -h 0'];
%group_train=int(group_train);
%group_sample=int(group_sample);
% trainXC_mean = mean(training); 
% trainXC_sd = sqrt(var(training)+0.01); 
% training1 = bsxfun(@rdivide, bsxfun(@minus, training, trainXC_mean), trainXC_sd); 
% sample1 = bsxfun(@rdivide, bsxfun(@minus, sample, trainXC_mean), trainXC_sd);

model = svmtrain(group_train,training,'-t 0 -h 0');
%[class,accTotal] = svmpredict(group_sample,sample,model);
[m,n]=size(sample);
predict_label=zeros(m,1);
for i=1:m
    %len=model.totalSV;
    %value=0;
    value=sum(model.sv_coef.*model.SVs.*sample(i,1));
%     for j=1:len
%         value=value+(model.sv_coef(j,1))*(model.SVs(j,1)*sample(i,1));
%     end
    value=value-model.rho;
    if value>0
        predict_label(i,1)=1;
    else
        predict_label(i,1)=0;
    end
end
sum(predict_label==group_sample)/size(group_sample,1)
%result(2,1)=result(2,1)+sum(predict_label==group_sample)/size(group_sample,1);
