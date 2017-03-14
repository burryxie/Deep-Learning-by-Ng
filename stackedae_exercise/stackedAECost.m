function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

z2 = stack{1}.w * data + repmat(stack{1}.b,1,size(data,2));
a2 = sigmoid(z2);
z3 = stack{2}.w * a2 + repmat(stack{2}.b,1,size(data,2));
a3 = sigmoid(z3);
z4 = softmaxTheta * a3;
a4 = exp(z4);
a4 = a4 ./ repmat(sum(a4,1),numClasses,1);
cost = -1/numClasses * sum(sum(a4 .* groundTruth)) + lambda/2 * sum(theta .^2);

softmaxThetaGrad = (-1/numClasses * (groundTruth - a4) * a3' + lambda * softmaxTheta);


% backpropogation
delta4 = -(softmaxTheta' * (groundTruth - a4)) .* a3 .* (1-a3);
delta3 = stack{2}.w' * delta4 .* a2 .* (1-a2);
%delta2 = stack{1}.w' * delta3 .* data .* (1-data);

stackgrad{1}.w = delta3 * data' / M;
stackgrad{1}.b = sum(delta3,2) / M;
stackgrad{2}.w = delta4 * a2' / M;
stackgrad{2}.b = sum(delta4,2) / M;




% n = numel(stack)
% z = cell(n+1, 1);
% a = cell(n+1, 1);
% a{1} = data;
% 
% 
% for l = (1:n)
%     z_temp = stack{l}.w * a{l};
%     z{l+1} = bsxfun(@plus, z_temp, stack{l}.b);
%     a{l+1} = sigmoid(z{l+1});
% end
% 
% % Equivalent of doing softmax calculation
% td = softmaxTheta * a{n+1};
% td = bsxfun(@minus, td, max(td));
% temp = exp(td);
% denominator = sum(temp);
% p = bsxfun(@rdivide, temp, denominator);
% 
% y = groundTruth;
% cost = (-1/M) * sum(sum(y .* log(p))) + (lambda / 2) * sum(sum(theta .^2));
% softmaxThetaGrad = (-1/M) * (y - p) * a{n+1}' + lambda * softmaxTheta;
% 
% % delta
% d = cell(n+1);
% d{n+1} = -(softmaxTheta' * (y - p)) .* a{n + 1} .* (1 -a{n + 1});
% 
% for l = (n:-1:2)
%     d{l} = stack{l}.w' * d{l+1} .* a{l} .* (1-a{l});
% end
% 
% for l = (n:-1:1)
%     stackgrad{l}.w = d{l+1} * a{l}' / M;
%     stackgrad{l}.b = sum(d{l+1}, 2) / M;
% end



% -------------------------------------------------------------------------
%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];
end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
