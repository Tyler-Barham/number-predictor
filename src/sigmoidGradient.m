function g = sigmoidGradient(z)
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

% Initialise g
g = zeros(size(z));

% Calulate the sigmoid gradient
g = sigmoid(z) .* (1 - sigmoid(z));

end
