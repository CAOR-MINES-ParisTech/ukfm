function [new_state, new_P] = ukf_jacobian_propagation(state, P, ...
    omega, f, dt, phi, phi_inv, cholQ, weights, idxs)
%UKF_JACOBIAN_PROPAGATION Propagate the UKF state in the situation where a
%part of the state is stationnary or parameters
%
% Syntax: [new_state, new_P] = ukf_propagation(state, P, omega, f, dt, ...
%     phi, phi_inv, cholQ, weights, idxs)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    omega - input, structure
%    f - propagation function, function
%    dt - integration step
%    phi - reduced retraction, function
%    phi_inv - reduced inverse retraction, function
%    cholQ - Cholewski decomposition of noise covariance matrix
%    weights - weight parameters of UKF
%    idxs - index of the error state in P that are propagating
%
% Outputs:
%    new_state - propagated state
%    new_P - propagated covariance matrix

TOL = 1e-9; % tolerance for assuring positivness of P

P_red = P(idxs, idxs);
F = eye(length(P));
% variable sizes
d = length(P_red);
q = length(cholQ);

P_red = P_red + TOL*eye(d);

% 1 - mean update
w = zeros(q, 1);
new_state = f(state, omega, w, dt);

% 2 - compute covariance w.r.t. state error
w_d = weights.d;

% set sigma points
xi_mat = w_d.sqrt_d_lambda * chol(P_red)';
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
Xi = w_d.wj * (xi_new_mat*[xi_mat -xi_mat]');
F(idxs, idxs) = (P_red\Xi')'; % Xi*P_red^{-1}
new_P = F*P*F';

% 3 - compute covariance w.r.t. noise
w_q = weights.q;
xi_new_mat = zeros(d, 2*q);

% rectract sigma points onto manifold
for j = 1:q
    w_plus = w_q.sqrt_q_lambda * cholQ(:, j);
    w_minus = -w_q.sqrt_q_lambda * cholQ(:, j);
    state_j_plus_new = f(state, omega, w_plus, dt);
    stat_j_minus_new = f(state, omega, w_minus, dt);
    xi_new_mat(:, j) = phi_inv(new_state, state_j_plus_new);
    xi_new_mat(:, q + j) = phi_inv(new_state, stat_j_minus_new);
end

% compute covariance
xi_mean = w_q.wj * sum(xi_new_mat, 2);
xi_new_mat = xi_new_mat - xi_mean;
Q = w_q.wj * (xi_new_mat*xi_new_mat') + w_q.wc0*(xi_mean*xi_mean');

% 4 - sum covariances
new_P(idxs, idxs) = new_P(idxs, idxs) + Q;
end