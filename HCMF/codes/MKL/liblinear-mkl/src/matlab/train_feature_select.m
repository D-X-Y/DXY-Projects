function [mklmodel] = train_feature_select(y,x,liblinear_param)
iter = 1;
[l n] = size(x);
d = ones(1,n) ./ n;
while true && iter < 100
	model = train(d',y,x,sprintf('%s',liblinear_param));
	w = model.w;
	d

	for i=1:n
		fh(i) = w(i)*w(i);
	end
	sumfh = sum(fh);
	for i=1:n
		newd(i) = fh(i) / sumfh;
	end

	if norm(newd - d,2) < 1e-3
		d = newd;
		break;
	end
	d = newd;
	input('');
end

mklmodel.model = model
mklmodel.d = d;
