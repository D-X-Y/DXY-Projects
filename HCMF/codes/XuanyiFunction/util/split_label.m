function [train_id, test_id] = split_label(label, ratio)

    train_id = [];
    ALL = unique(label);
    for index = 1:numel(ALL)
        total = find(label == ALL(index));
        number = numel(total);
        ID = randperm(number);
        train_id = [train_id; total(ID(1:ceil(number*ratio)))];
    end
    test_id = setdiff((1:numel(label)), train_id);
end
