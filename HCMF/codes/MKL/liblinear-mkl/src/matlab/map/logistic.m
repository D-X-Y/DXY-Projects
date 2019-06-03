function [newx] = logistic(x)
newx = 1 ./ (1 + exp(-x));

