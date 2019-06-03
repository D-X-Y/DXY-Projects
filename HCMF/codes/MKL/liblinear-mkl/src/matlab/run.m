[y,xt] = libsvmread('../heart_scale');
[l n] = size(xt);
d = ones(n,1);
model=train(d, y, xt, '-s 3');
[l,a]=predict(y, xt, model);

