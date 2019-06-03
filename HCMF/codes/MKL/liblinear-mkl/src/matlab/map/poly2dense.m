function [newx] = poly2dense(x)
x = full(x);
[l n] = size(x);
newx = zeros(l,n*(n+1)/2);
k = 1;
for i=1:n
	for j=i:n
		if i==j
			newx(:,k) = x(:,i).*x(:,j);
		else
			newx(:,k) = sqrt(2).*x(:,i).*x(:,j);
		end
		k = k + 1;
	end
end
newx = [newx x];
