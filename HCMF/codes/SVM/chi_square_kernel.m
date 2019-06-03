function kernel_matrix = chi_square_kernel(vectorA, vectorB)
%computes chi-square kernel for svm

size(vectorA, 2) == size(vectorB, 2);
[na_vec, dim]=size(vectorA);
[nb_vec, ~  ]=size(vectorB);
kernel_matrix=zeros(na_vec, nb_vec);
vectorA = full(vectorA);
vectorB = full(vectorB);

kmatrix = cell(na_vec, 1);
parfor i=1:na_vec
    vec_i = vectorA(i, :);
    kmatrix{i} = zeros(1, nb_vec);
    for j=1:nb_vec
        val = chi_square_val(vec_i, vectorB(j,:));
        kmatrix{i}(j) = val;
    end
end
for i = 1:na_vec
    for j = 1:nb_vec
        kernel_matrix(i,j) = kmatrix{i}(j);
    end
end
kernel_matrix=[(1:na_vec)' , kernel_matrix];
end

function chi_val=chi_square_val(vec1,vec2)
% 1-sum( (xi-xj)^2/0.5*(xi+xj) )
chi_val=1 - 2*(vec1-vec2)./(vec1+vec2+eps)*(vec1-vec2)';
%chi_val=1-chi_val;
%chi_val=exp(chi_val);
% a=(vec1-vec2).^2;
% b=(vec1+vec2);
% resHelper = 1-sum(2*a./(b + eps));
end

