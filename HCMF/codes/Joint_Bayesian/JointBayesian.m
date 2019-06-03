function [mappedX, mapping] = JointBayesian(X, labels)
%LDA Perform the LDA algorithm
%
%   [mappedX, mapping] = lda(X, labels, no_dims)
%
% The function runs LDA on a set of datapoints X. The variable
% no_dims sets the number of dimensions of the feature points in the 
% embedded feature space (no_dims >= 1, default = 2). The maximum number 
% for no_dims is the number of classes in your data minus 1. 
% The function returns the coordinates of the low-dimensional data in 
% mappedX. Furthermore, it returns information on the mapping in mapping.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology
% Make sure data is zero mean
%     mapping.mean = mean(X, 1);
%     [COEFF,SCORE] = princomp(X,'econ');
%     X = SCORE(:,1:400);
%     X = bsxfun(@minus,X,mapping.mean);
  max_buffer_number = 20000;
  max_iteration = 500;

  m = length(labels);
  n = size(X,2);
	
	% Make sure labels are nice
	[classes, bar, labels] = unique(labels);
  nc = length(classes);
	
	% Intialize Sw
	Sw = zeros(size(X, 2), size(X, 2));
    
  cur = {};
  withinCount = 0;
  numberBuff = zeros(max_buffer_number, 1);
% numberInvert = zeros(1000,1);
    maxNumberInOneClass = [];
    for i=1:nc
        % Get all instances with class i
        cur{i} = X(labels == i,:);
        if size(cur{i},1)>1
            withinCount = withinCount + size(cur{i},1);
        end;
        if numberBuff(size(cur{i},1)) ==0
            numberBuff(size(cur{i},1)) = 1;
            maxNumberInOneClass = [maxNumberInOneClass;size(cur{i},1)];
        end;
    end;
    disp([nc withinCount]);
    fprintf('prepare done, maxNumberInOneClass=%d.\n',length(maxNumberInOneClass));
    tic;
    u = zeros(n,nc);
    ep = zeros(n,withinCount);
    nowp = 1;
	% Sum over classes
	for i=1:nc
		% Update within-class scatter
        u(:,i) = mean(cur{i},1)';
        
        if size(cur{i},1)>1
            ep(:,nowp:nowp+ size(cur{i}, 1)-1) = bsxfun(@minus,cur{i}',u(:,i));
            nowp = nowp + size(cur{i}, 1);
%             C = cov(cur{i});
%             p = size(cur{i}, 1) / withinCount;
%             Sw = Sw + (p * C);
        end;
	end;
        Su = cov(u');
        Sw = cov(ep');
%     Su = u*u'/nc;
%     Sw = ep*ep'/withinCount;
%     Su = cov(rand(n,n));
%     Sw = cov(rand(5*n,n));
    toc;
%     F = inv(Sw);
%     mapping.Su = Su;
%     mapping.Sw = Sw;
%     mapping.G = -1 .* (2 * Su + Sw) \ Su / Sw;
%     mapping.A = inv(Su + Sw) - (F + mapping.G);
%     mappedX = X;
% end
    
    oldSw = Sw;
%     Gs = cell(maxNumberInOneClass,1);
    SuFG = cell(20000,1);
    SwG = cell(20000,1);
    
    for l=1:max_iteration
%         tic;
        F = inv(Sw);
        ep =zeros(n,m);
        nowp = 1;
        for g = 1:max_buffer_number
            if numberBuff(g)==1
                G = -1 .* (g .* Su + Sw) \ Su / Sw;
                SuFG{g} = Su * (F + g.*G);
                SwG{g} = Sw*G;
            end;
        end;
        for i=1:nc
            nnc = size(cur{i}, 1);
%             G = Gs{nnc};
            u(:,i) = sum(SuFG{nnc} * cur{i}',2);
            ep(:,nowp:nowp+ size(cur{i}, 1)-1) = bsxfun(@plus,cur{i}',sum(SwG{nnc}*cur{i}',2));
            nowp = nowp+ nnc;
        end;
        Su = cov(u');
        Sw = cov(ep');
%     Su = u*u'/nc;
%     Sw = ep*ep'/withinCount;
        fprintf('%d %f\n',l,norm(Sw - oldSw)/norm(Sw));
%         toc;
        if norm(Sw - oldSw)/norm(Sw)<1e-5
            break;
        end;
        oldSw = Sw;
    end;
    F = inv(Sw);
    mapping.G = -1 .* (2 * Su + Sw) \ Su / Sw;
    mapping.A = inv(Su + Sw) - (F + mapping.G);
    mapping.Sw = Sw;
    mapping.Su = Su;
%     mapping.U = chol(-G,'upper');
%     mapping.COEFF = COEFF;
%     mapping.y = mapping.U * X';
    mapping.c = zeros(m,1);
    for i = 1:m
        mapping.c(i) = X(i,:) * mapping.A * X(i,:)';
    end;
    mappedX = X;
   
