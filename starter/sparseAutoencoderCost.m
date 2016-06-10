function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% add 1's to accomodate extra bias term in wb matrices
m = size(data,2);
data = [data; ones(1,size(data,2))];

%-------FORWARD--------
% add bias to weight matrix
wb1 = [W1 b1]; % horizontal concat
wb2 = [W2 b2];

% move to hidden layer
z1 = wb1*data;
a = sigmoid(z1);

% move to final layer
z2 = wb2*[a; ones(1,size(a,2))]; % add ones again to accomodate bias term

h = sigmoid(z2); % output
cost = (1/m)*(sum(sum(0.5*((h-data(1:visibleSize,:)).^2)))); % error

%-------BACKWARD--------
% "error term" for how "responsible" a node is to the output
% using just the weights, not the bias terms
s2 = -(data(1:visibleSize,:)-h).*(h.*(1-h));
s1 = (wb2(:,1:hiddenSize)'*s2).*(a.*(1-a));

% partial derivatives, or gradients, summed across samples during matrix
% operation.
d2 = s2*[a; ones(1,size(a,2))]'; % add ones again to accomodate bias term; % for weights moving from in hidden to output (W2)
d1 = s1*data'; % for weights movingrom input to hidden (W1)

% take the average partial derivatives for this iteration of wb1 and wb2,
% because this is a full gradient descent. If it was a single sample at a
% time, the gradient would be updated with each iteration, without
% averaging since the weights are updated with each iteration. Here, the
% weights are not being updated, so we would like to know what the general
% direction should be.
b2grad = (1/m)*d2(:,end);
b1grad = (1/m)*d1(:,end);

W2grad = (1/m)*(d2(:,1:hiddenSize)) + lambda*wb2(:,1:hiddenSize);
W1grad = (1/m)*(d1(:,1:visibleSize)) + lambda*wb1(:,1:visibleSize);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

