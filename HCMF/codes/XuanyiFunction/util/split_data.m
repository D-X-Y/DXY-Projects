function [train_id, test_id] = split_data(number, ratio)
    total = randperm(number);
    train = ceil(number*ratio);
    %test  = number - train;
    train_id = total(1: train)';
    test_id  = total(train+1: end)';
end
