function prob = ConvertDec2Pro(dec_values, K)
    assert(size(dec_values,2) == K*(K-1)/2);
    N = size(dec_values,1);
    prob = zeros(N, K);
    for index = 1:N
        prob(index,:) = Sub2Prob(dec_values(index,:), K);
    end
end

function prob = Sub2Prob(dec_value, K)

    %P = zeros(K, K);
    prob = zeros(K,1);
    k = 1;
    for i = 1:K
        for j = i+1:K
            %P(i,j) = dec_value(k);
            %P(j,i) = 1 - dec_value(k);
            if (dec_value(k)>0)
                prob(i) = prob(i) + 1;
            else
                prob(j) = prob(j) + 1;
            end
            k = k + 1;
        end
    end
end
