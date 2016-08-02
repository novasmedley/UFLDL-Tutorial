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
% labels: A vector contameining labels, where labels(i) is the label for the
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

% --------- FORWARD ---------
% AE1 gave us some pretrained weights for what W1, b1, W2, b2, and softmax
% thetas matrix (labels-by-inputSize) should be. Here, push those weights
% forward from the input data.

% combine weights and bias
wb1 = [stack{1}.w stack{1}.b]; %horz concat
wb2 = [stack{2}.w stack{2}.b];

% get layer 2 activation values
z2 = wb1*[data; ones(1,size(data,2))];  % accomodate bias term in data matrix
a2 = sigmoid(z2);

% get layer 3 activation values
z3 = wb2*[a2; ones(1,size(a2,2))];   % accomodate bias term in data matrix
a3 = sigmoid(z3);

% --------- softmax COST and gradient and class probabilities ---------

[cost softmaxThetaGrad, prob] = softmaxCost(softmaxTheta, numClasses, hiddenSize, lambda, a3, labels);

% --------- BACKWARD ---------
% get "error terms" for layers
s3 = ( -1*softmaxTheta'*( groundTruth-prob ) ).*( a3.*(1-a3) );
s2 = ( stack{2}.w'*s3 ).*( a2.*(1-a2) );

% get partial derivatives
d2 = s3*[a2; ones(1,size(a2,2))]';
d1 = s2*[data; ones(1,size(data,2))]';

stackgrad{2}.b = (1/M)*d2(:,end);
stackgrad{1}.b = (1/M)*d1(:,end);

sizeOfLayer2 = size(stack{2}.w,2);
stackgrad{2}.w = (1/M)*d2(:,1:sizeOfLayer2) ;%+ lambda*wb2(:,1:sizeOfLayer2); NOT REGULARIZED IN STACKED!!
stackgrad{1}.w = (1/M)*d1(:,1:inputSize) ;%+ lambda*wb1(:,1:inputSize);

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
