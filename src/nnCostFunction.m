function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Add bias to X
a1 = [ones(m, 1) X];

% Get a2
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% Add bias to a2
a2 = [ones(m, 1) a2];

% Get a3
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Convert y to a binary matrix representation
ybinary = full(sparse(1:rows (y), y, 1));

% Calculate the cost
cost = (1 / m) * sum( sum( (-ybinary .* log(a3)) - (1 - ybinary) .* log(1 - a3) ) );

% Calculate the regularization
reg = (lambda / (2 * m)) * ( sum( sum(Theta1(:, 2:end).^2) ) + sum( sum(Theta2(:, 2:end).^2) ) );

% Cost + regularized part
J = cost + reg;

% Calulate the deltas
d3 = a3 - ybinary;
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2);
delta2 = d3' * a2;
delta1 = d2' * a1;

% Store theta gradients
Theta1_grad = (1 / m) * (delta1 + lambda * Theta1);
Theta2_grad = (1 / m) * (delta2 + lambda * Theta2);

% Recalulate the bias'
Theta1_grad(:, 1) = (1 / m) * delta1(:, 1);
Theta2_grad(:, 1) = (1 / m) * delta2(:, 1);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
