clear;
ntimes = 5;

fp = fopen('result.html','w');
fprintf(fp,'<html>\n<body>\nUsing Mappings : f(x) = x, f(x) = x.^2, f(x) = x.^3 ... f(x) = x.^30 as base mapping functions<br/>\n<table border=1>');

fprintf(fp,'<tr>');
fprintf(fp,'<td></td>');
fprintf(fp,'<td>%d times accuracy mean +- stdev of liblinear linear feature</td>',ntimes);
fprintf(fp,'<td>avg time</td>');
fprintf(fp,'<td>%d times accuracy mean +- stdev of liblinear MKL</td>',ntimes);
fprintf(fp,'<td>avg time</td>');
fprintf(fp,'<td>%d times accuracy mean +- stdev of direct feature combination</td>',ntimes);
fprintf(fp,'<td>avg time</td>');
fprintf(fp,'</tr>\n');
%for data={'/tmp/covtype.libsvm.binary.scale','../heart_scale','../ionosphere_scale','../diabetes_scale','../breast-cancer_scale','../sonar_scale'}
	%data={'../dna.scale','../glass.scale','../heart_scale','../ionosphere_scale','../diabetes_scale','../breast-cancer_scale','../sonar_scale'}
for data = {'/home/dongxuanyi/AAAI/heart_scale'};
	clear newx
	clear func
	tic;
	data = char(data);
	[y x] = libsvmread(data);
	[l n] = size(x);
	for j=1:3
		newx(:,(j-1)*n+1:j*n) = x.^j;
		func{j} = inline(sprintf('x.^%d',j));
	end
	preprocess_time = toc;

	for i=1:ntimes
		subidx = randsample(l,ceil(l*4/5));
		predictidx = setdiff(1:l,subidx);
		yt = y(subidx);
		xt = x(subidx,:);
		[l n] = size(x);
	
		newx = sparse(newx);

		%cross validation
		bestacc = -1;
		bestc = 0;
		for log2c=-3:3
			c = 2^log2c;
			acc= train(ones(size(x,2),1),yt,x(subidx,:),sprintf('-s 3 -v 5 -c %f',c));
			if acc > bestacc
				bestacc = acc;
				bestc = c;
			end
		end
		%linear svm -- best parameter!
		tic;
		model = train(ones(size(x,2),1),yt,x(subidx,:),sprintf('-s  3 -c %f',bestc));
		[py acc devs] = predict(y(predictidx,:),x(predictidx,:),model);
		linear_acc(i) = acc;
		linear_time(i) = toc;

		%cross validation
		bestacc = -1;
		bestc = 0;
		for log2c=-3:3
			c = 2^log2c;
			acc= train(ones(size(newx,2),1),yt,newx(subidx,:),sprintf('-s 3 -v 5 -c %f',c));
			if acc > bestacc
				bestacc = acc;
				bestc = c;
			end
		end
		%linear svm + combine -- best parameter!
		tic;
		model = train(ones(size(newx,2),1),yt,newx(subidx,:),sprintf('-s 3 -c %f',bestc));
		[py acc devs] = predict(y(predictidx,:),newx(predictidx,:),model);
		orig_acc(i) = acc;
		orig_time(i) = toc;

		%cross validation
		bestacc = -1;
		bestc = 0;
		for log2c=-3:3
			c = 2^log2c;
			acc= train_mkl(yt,xt,func,sprintf('-s 3 -c %f',c),'',5);
			if acc > bestacc
				bestacc = acc;
				bestc = c;
			end
		end
		%linear svm + mkl -- best parameter!
		tic;
		mklmodel = train_mkl(yt,xt,func,sprintf('-s 3 -c %f',bestc));
		[py acc devs] = predict_mkl(y(predictidx,:),x(predictidx,:),mklmodel);
		mkl_acc(i) = acc;
		mkl_time(i) = toc;
%		input('')
	end

%	disp(sprintf('\nResult of data set %s\n',data))
%	disp('Mean accuracy via selecting linear combination of mappings phi(x) = x and phi(x) = x.^2 (Primal MKL)');
%	acc = sumacc / 5
%	disp('Mean accuracy via concatenating features genereated by mappings phi(x) = x and phi(x) = x.^2');
%	acc_orig = sumacc_orig / 5

	fprintf(fp,'<tr>');
	fprintf(fp,'<td>%s</td>',data);
	fprintf(fp,'<td>%.4f +- %.4f</td>',mean(linear_acc),std(linear_acc));
	fprintf(fp,'<td>%.4f</td>',mean(linear_time));
	fprintf(fp,'<td>%.4f +- %.4f</td>',mean(mkl_acc),std(mkl_acc));
	fprintf(fp,'<td>%.4f</td>',mean(mkl_time));
	fprintf(fp,'<td>%.4f +- %.4f</td>',mean(orig_acc),std(orig_acc));
	fprintf(fp,'<td>%.4f</td>',mean(orig_time) + preprocess_time);
	fprintf(fp,'</tr>\n');
end
fprintf(fp,'</table>\n</body>\n</html>');
fclose(fp);
