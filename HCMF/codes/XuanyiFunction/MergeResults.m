function X = MergeResults(results, label, norm_id)
    M = numel(results);
    X = zeros(size(results{1}));
    for i = 1:M
        X = X + NORM(results{i}, norm_id);
    end
    X = X / M;
    if (numel(label) ~= 0)
        assert(size(X,1) == size(label,1));
        [mx, ids] = max(X, [], 2);
        fprintf('Accuracy : %.3f\n', sum(ids==label) / numel(ids));
    end
end

function ANS = NORM(result, norm_id)
    if (norm_id == 0)
        ANS = result;
    elseif (norm_id == 1)
        prob_estimates = result;
        probs = repmat(sum(prob_estimates .* prob_estimates, 2), 1, size(prob_estimates,2));
        ANS = prob_estimates ./ probs;
    elseif (norm_id == 2)
        prob_estimates = result;
        probs = exp(prob_estimates); 
        ANS = probs ./ repmat(sum(probs, 2), 1, size(prob_estimates,2));
    else
        assert false
    end
end
