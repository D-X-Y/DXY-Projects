function [py acc decv] = predict_mkl(y,x,mklmodel)
phis = mklmodel.phis;
m = length(phis);

for i=1:m
	phi = phis{i};
	xadd = phi(x);
	if i == 1
		fstart(i) = 1;
		fend(i) = size(xadd,2);
	else
		fstart(i) = fend(i-1) + 1;
		fend(i) = fend(i-1) + size(xadd,2);
	end
	xs(:,fstart(i):fend(i)) = xadd;
end

if length(mklmodel.classes) <= 2
	[py acc decv] = predict(y,xs,mklmodel.model);
else
	classes = mklmodel.classes;
	for j = 1:length(classes)
		[py_ acc_ decv_] = predict(y,xs,mklmodel.model{j});
		py(:,j) = py_;
		decv(:,j) = decv_ .*  mklmodel.model{j}.Label(1);
	end
	[dummy idxes] = max(decv');
	py = classes(idxes);
	acc = sum(y==py) / length(y) * 100;
end
