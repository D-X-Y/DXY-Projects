function acc = CrossValidation(train_scores, train_label, test_scores, test_label, verbose)
    NU = 0.5:0.1:0.9;
%    NU(end+1) = 0.95;
%{
    split = SPLIT(train_label);
    M = numel(train_scores);
    best_los = inf;
    best_nu  = 0.05;
for nn = 1:numel(NU)
    nu = NU(nn);
    ave = zeros(numel(split), 1);
    for i = 1:numel(split)
        current_test_id = split{i};
        current_train_id = [];
        for j = 1:numel(split)
            if (i ~= j)
                current_train_id = [current_train_id; split{j}];
            end
        end
        c_train_scores = cell(M, 1);
        c_test_scores  = cell(M, 1);
        c_train_label  = train_label(current_train_id);
        c_test_label   = train_label(current_test_id);
        for j = 1:M
            c_train_scores{j} = train_scores{j}(current_train_id, :);
            c_test_scores{j} = train_scores{j}(current_test_id, :);
        end
        [~, acc, los] = LPBoost(c_train_scores, c_train_label, c_test_scores, c_test_label, nu, verbose);
        ave(i) = los;
    end
    if (mean(ave) < best_los)
        best_nu = nu;
        best_los = mean(ave);
    end
    [~, acc, los] = LPBoost(train_scores, train_label, test_scores, test_label, nu, verbose);
    if (verbose == 1)
        fprintf('nu : %.2f, acc : %.3f, loss : %.3f \n', nu, acc, los); 
    end
end
    if (verbose == 1)
        fprintf('Best nu : %.4f, Best loss : %.4f\n', best_nu, best_los);
    end
    [~, acc, ~] = LPBoost(train_scores, train_label, test_scores, test_label, best_nu, verbose);
%}
disp(NU);
Tacc = zeros(numel(NU), 1);
for nn = 1:numel(NU)
    nu = NU(nn);
    tic;
    [~, Tacc(nn), ~] = LPBoost(train_scores, train_label, test_scores, test_label, nu, 1);
    toc
end
%acc = sort(unique(Tacc));
%acc = acc(max(1,end-1));
acc  = max(Tacc);

end

function split = SPLIT(train_label)
    total = numel(train_label);
    idx = randperm(total)';
    tot = 5;
    num = ceil(total/tot);
    numers = 0;
    for i = 1:tot
        numers(end+1) = min(num, total - numers(end));
        numers(end) = numers(end) + numers(end-1);
    end
    split = cell(tot, 1);
    for i = 2:tot+1
        split{i-1} = idx(numers(i-1)+1:numers(i));
    end
end
