function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
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

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Fully vectorized code is wrong!!
%X = [ones(m, 1), X];
%Z2 = X * Theta1';
%A2 = sigmoid(Z2);

%Z3 = [ones(size(A2, 1), 1) A2] * Theta2';
%A3 = sigmoid(Z3);

%A3 is a m X 10 matrix
%y is a m X 1 matrix. Convert y to m*10 matrix
%Y = zeros(num_labels, m);

%Y(sub2ind(size(Y), y', 1:m)) = 1;

%error = Y' .* log(A3) + (1 - Y') .* log(1 - A3);
%J = -sum(error)/m;

%Theta1Squared = sum(Theta1 .* Theta1);
%Theta2Squared = sum(Theta2 .* Theta2);

%J = J + lambda/(2*m) * (sum(Theta1Squared(2:end)) + sum(Theta2Squared(2:end)))

% Theta1 : 25x401
% Theta2 : 10x26

% Half vectorized code
for i = 1:m
    Xi = [1, X(i,:)]; % 1x401
    Z2 = Xi * Theta1'; % 1x25
    A2 = sigmoid(Z2); % 1x25
    Z3 = [1, A2] * Theta2'; % 1x10
    A3 = sigmoid(Z3); % 1x10

    Y = zeros(1, num_labels);
    Y(y(i)) = 1;

    D3 = A3 - Y; % 1x10
    D2 = (D3 * Theta2(:, 2:end)) .* sigmoidGradient(Z2); % 1x25
    Theta2_grad = Theta2_grad + D3' * [1, A2]; % 10x26
    Theta1_grad = Theta1_grad + D2' * Xi; % 25x401

    J = J + (-1/m) * sum(Y .* log(A3) + (1 - Y) .* log (1 - A3));

end

% Adding regularization
J = J + lambda/(2*m) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));
Theta1_grad = Theta1_grad/m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);

Theta2_grad = Theta2_grad/m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
