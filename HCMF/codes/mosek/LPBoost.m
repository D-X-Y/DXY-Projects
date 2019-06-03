function [beta, accuracy, loss] = LPBoost(train_scores, train_label, test_scores, test_label, nu, verbose)
    % \rho, \beta, \xi to minimize function : -\nu + \sigma(\xi) / \nu / N
    % for mosek function linprog. 
    M = numel(train_scores); assert( M == numel(test_scores));
    N = numel(train_label );
    K = size(train_scores{1}, 2); assert(numel(unique(train_label)) == K);
    for i = 1:M, assert(size(train_scores{i},1) == N && size(test_scores{i},1) == numel(test_label) && size(test_scores{i},2) == K); end

    % [x,fval,exitflag,output,lambda] 
    %  = linprog(f,A,b,B,c,l,u,x0,options) 
    %% rho unconstrained, xi >= 0 constraints
    varLB = -Inf(1 + N + M, 1);
    varUB =  Inf(1 + N + M, 1);
    for n = 1:N+M, varLB(1+n) = 0; end
    %% Setup objective function: min -\rho + D \sum_i \xi_i
    objective = zeros(1 + N + M, 1);
    objective(1) = -1;
    slack_penalty = 1 / (nu * N);
    for n = 1:N, objective(1+n) = slack_penalty; end
    %% set B*X = c, \sigma(\beta) = 1
    B = zeros(1, 1 + N + M);
    for m = 1:M, B(1+N+m) = 1; end;
    c = 1;
    
    data_pepare = tic;
    %% set A*X <= b, b = 0
    A = sparse(N*(K-1), 1 + N);
    INDEX_I = zeros(N*(K-1), 1);
    INDEX_J = zeros(N*(K-1), 1);
    %%Temp_Train_Scores = cat(3, train_scores{:});
    cols = 0;
    for i = 1:N
        class = train_label(i);
        for j = 1:K
            if(j==class),continue;end
            cols = cols + 1;
            A(cols, 1+i) = -1;
            INDEX_I(cols) = i;
            INDEX_J(cols) = j;
        %   for m = 1:M
        %       A(cols, 1+N+m) = train_scores{m}(i,j) - train_scores{m}(i,class);
        %   end
        end
        if (rem(i, 2000) == 0)
            fprintf('%d / %d , Nu : %.3f, Data Prepare Time : %.3f \n', i, N, nu, toc(data_pepare));
        end
    end
    AA = zeros(N*(K-1), M);
    Temp_Train_Scores = cat(3, train_scores{:});
    Temp_Train_Scores = permute(Temp_Train_Scores, [3,1,2]);
    parfor cols = 1:(K-1)*N
%   for cols = 1:(K-1)*N
        i = INDEX_I(cols);
        j = INDEX_J(cols);
        class = train_label(i);
        A(cols, 1) = 1;
        AA(cols, :) = Temp_Train_Scores(:,i,j) - Temp_Train_Scores(:,i,class);
    end
    A = [A, sparse(AA)];
    cols = 0;
    b = zeros(N*(K-1), 1);
    data_pepare = toc(data_pepare);

    mosek_begin = tic;
    [x,fval,exitflag,output,lambda] = linprog(objective, A, b, B, c, varLB, varUB, [], []);
    mosek_time = toc(mosek_begin);
    if (exitflag < 0), fprintf('The problem is likely to be either primal or dual infeasible.!!!!\n');end
    beta = x(1+N+1:end);
    Final_Scores = zeros(numel(test_label), K);
    for i = 1:M
        Final_Scores = Final_Scores + test_scores{i} * beta(i);
    end
    [~, idx] = max(Final_Scores, [], 2);
    accuracy = mean(idx==test_label);
    loss     = 0;
    for i = 1:size(Final_Scores,1)
        loss = loss + LOSS(Final_Scores(i,:), test_label(i));
    end
    loss = loss / size(Final_Scores,1);
    if (verbose == 1)
        fprintf('nu: %.2f, accuracy : %.5f, loss : %.3f, data time : %.2f, mosek time : %.2f\n', nu, accuracy, loss, data_pepare, mosek_time);
    end
end

function loss = LOSS(vector, label)
    vector = exp(vector);
    vector = vector / sum(vector);
    loss = -log(vector(label));
end
