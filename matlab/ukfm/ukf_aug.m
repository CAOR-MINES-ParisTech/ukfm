function [aug_state, aug_P] = ukf_aug(state, P, y, z, z_aug, phi, ...,
    weights, idxs, R)
%UKF_AUG Augment the state of the UKF
%
% Syntax: [aug_state, aug_P] = ukf_aug(state, P, y, z, z_aug, phi, ...,
%    weights, idxs, R)
% Inputs:
%    state - state
%    P - covariance matri
%    y - measurement
%    z - new part of the state function, function
%    z_aug - new state function, function
%    phi - retraction, function
%    weights - weight parameters of UKF
%    idxs - index of the error state in P that are present in measurement
%    R - measurement covariance
%
% Outputs:
%    aug_state - augmented state
%    aug_P - augmented covariance matrix

TOL = 1e-9; % tolerance for assuring positivness of P

P_red = P(idxs, idxs);
% set variable size
d = length(P_red);
q = length(R);

P_red = P_red + TOL*eye(d);

% 1 - augment state mean
aug_state = z_aug(state, y);
z_hat = z(state, y);

% 2 - compute Jacobian and covariance from state
% set sigma points w.r.t. state
w_d = weights.d;
xi_mat = w_d.sqrt_d_lambda * chol(P_red)';

% compute measurement sigma_points
z_mat = zeros(q, 2*d);
for j = 1:d
    state_j_plus = phi(state, xi_mat(:, j));
    state_j_minus = phi(state, -xi_mat(:, j));
    z_mat(:, j) = z(state_j_plus, y);
    z_mat(:, d + j) = z(state_j_minus, y);
end

% measurement mean
z_bar = w_d.wm0 * z_hat + w_d.wj * sum(z_mat, 2);

% prune mean before computing covariance
z_mat = z_mat - z_bar;
P_state = w_d.wj * (z_mat*z_mat') + w_d.wc0*((z_bar-z_hat)*(z_bar-z_hat)');

Xi = w_d.wj*z_mat*[xi_mat -xi_mat]';
H = zeros(q, length(P));
H(:, idxs) = (P_red\Xi')'; % Xi*P_red^{-1}

% 3 - compute covariance from measurement
% set sigma points w.r.t. noise
w_q = weights.q;
y_mat = w_q.sqrt_q_lambda * chol(R)';

% compute measurement sigma_points
z_mat = zeros(q, 2*q);
for j = 1:q
    y_j_plus = y + y_mat(:, j);
    y_j_minus = y - y_mat(:, j);
    z_mat(:, j) = z(state, y_j_plus);
    z_mat(:, q + j) = z(state, y_j_minus);
end

% measurement mean
z_bar = w_q.wm0 * z_hat + w_q.wj * sum(z_mat, 2);

% prune mean before computing covariance
z_mat = z_mat - z_bar;
P_mes = w_q.wj * (z_mat*z_mat') + w_q.wc0*((z_bar-z_hat)*(z_bar-z_hat)');

% 4 - compute augmented covariance
P_aug_aug = P_state + P_mes;
P_aug_prev = H*P;
aug_P = [P P_aug_prev';
    P_aug_prev P_aug_aug];
end