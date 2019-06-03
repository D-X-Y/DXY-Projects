function I = Convert2Image(X, rect_len)
    M = numel(X);
    [N, K] = size(X{1});
    %gaps = [2,3];
    gaps = rect_len;

    I = uint8(zeros(N*rect_len(2) + gaps(1) * 2, M*K*rect_len(1) + gaps(2) * (M+1), 3)) + 255;
    fprintf('image size : %d, %d\n', size(I,1), size(I,2));

    RGBs = uint8( jet(K) * 255 );

    for i = 1:M
       assert(all([N,K] == size(X{i})));
       RR = i * K * rect_len(1) + i * gaps(2);
       LL = RR - K * rect_len(1) + 1;
       for j = 1:N
           [~,id] = max(X{i}(j,:));
           R = j * rect_len(2) + gaps(1);
           L = R - rect_len(2) + 1;
           rgb = reshape( RGBs(id,:), 1, 1, 3);
           I(L:R, LL:RR,:) = repmat(rgb, R-L+1, RR-LL+1, 1);
       end
    end

end
