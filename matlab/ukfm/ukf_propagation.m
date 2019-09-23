function [new_state, new_P] = ukf_propagation(state, P, omega, f, dt, ...
    phi, phi_inv, cholQ, weights)
%UKF_PROPAGATION Propagate the UKF state
%
% Syntax: [new_state, new_P] = ukf_propagation(state, P, omega, f, dt, ...
%    phi, phi_inv, cholQ, weights)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    omega - input, structure
%    f - propagation function, function
%    dt - integration step
%    phi - retraction, function
%    phi_inv - inverse retraction, function
%    cholQ - Cholewski decomposition of noise covariance matrix
%    weights - weight parameters
%
% Outputs:
%    new_state - propagated state
%    new_P - propagated covariance matrix

TOL = 1e-9; % tolerance for assuring positivness of P

% variable sizes
d = length(P);
q = length(cholQ);

P = P + TOL*eye(d);

% 1 - update mean
w = zeros(q, 1);
new_state = f(state, omega, w, dt);

% 2 - compute covariance w.r.t. state error
w_d = weights.d;

% set sigma points
xi_mat = w_d.sqrt_d_lambda * chol(P)';
xi_new_mat = zeros(d, 2*d);

% rectract sigma points onto manifold
for j = 1:d
    state_j_plus = phi(state, xi_mat(:, j));
    state_j_minus = phi(state, -xi_mat(:, j));
    state_j_plus_new = f(state_j_plus, omega, w, dt);
    state_j_minus_new = f(state_j_minus, omega, w, dt);
    xi_new_mat(:, j) = phi_inv(new_state, state_j_plus_new);
    xi_new_mat(:, d + j) = phi_inv(new_state, state_j_minus_new);
end

% compute covariance
xi_mean = w_d.wj * sum(xi_new_mat, 2);
xi_new_mat = xi_new_mat - xi_mean;
new_P = w_d.wj * (xi_new_mat*xi_new_mat') + w_d.wc0*(xi_mean*xi_mean');

% 3 - compute covariance w.r.t. noise
w_q = weights.q;
xi_new_mat = zeros(d, 2*q);

% rectract sigma points onto manifold
for j = 1:q
    w_plus = w_q.sqrt_q_lambda * cholQ(:, j);
    w_minus = -w_q.sqrt_q_lambda * cholQ(:, j);
    state_j_plus_new = f(state, omega, w_plus, dt);
    state_j_minus_new = f(state, omega, w_minus, dt);
    xi_new_mat(:, j) = phi_inv(new_state, state_j_plus_new);
    xi_new_mat(:, q + j) = phi_inv(new_state, state_j_minus_new);
end

% compute covariance
xi_mean = w_q.wj * sum(xi_new_mat, 2);
xi_new_mat = xi_new_mat - xi_mean;
Q = w_q.wj * (xi_new_mat*xi_new_mat') + w_q.wc0*(xi_mean*xi_mean');

% 3 - sum covariances
new_P = new_P + Q;
end