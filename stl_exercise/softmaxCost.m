function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2); %N, number of samples

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
cur_res = exp(theta * data);
cur_res_colsum = sum(cur_res,1);
cur_res_colsum = repmat(cur_res_colsum,numClasses,1);
cur_res = cur_res./cur_res_colsum;
cur_res = log(cur_res);
cost = -1./numCases * sum(sum(groundTruth .* cur_res)) + lambda/2 * sum(sum(theta.^2));
thetagrad = -1./numCases * ((groundTruth - exp(cur_res)) * data') + lambda * theta;



% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

