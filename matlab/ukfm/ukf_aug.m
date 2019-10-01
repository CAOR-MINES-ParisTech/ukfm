function [aug_state, aug_P] = ukf_aug(state, P, y, z_aug, phi, ...
    aug_phi_inv, weights, idxs, R)
%UKF_AUG Augment the state of the UKF
%
% Syntax: [aug_state, aug_P] = ukf_aug(state, P, y, z_aug, phi, ...,
%    aug_phi_inv, weights, idxs, R)
%
% Inputs:
%    state - state
%    P - covariance matri
%    y - measurement
%    z_aug - new state function, function
%    phi - retraction, function
%    aug_phi_inv - inverse retraction for new state, function
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

% augment state mean
aug_state = z_aug(state, y);

% compute Jacobian and covariance from state
% set sigma points w.r.t. state
w_d = weights.d;
xis = w_d.sqrt_d_lambda * chol(P_red)';

% compute measurement sigma_points
zs = zeros(q, 2*d);
for j = 1:d
    s_j_p = phi(state, xis(:, j));
    s_j_m = phi(state, -xis(:, j));
    z_j_p = z_aug(s_j_p, y);
    z_j_m = z_aug(s_j_m, y);
    zs(:, j) = aug_phi_inv(aug_state, z_j_p);
    zs(:, d + j) = aug_phi_inv(aug_state, z_j_m);
end

% measurement mean
z_bar = w_d.wj * sum(zs, 2);

% prune mean before computing covariance
zs = zs - z_bar;
P_state = w_d.wj * (zs*zs') + w_d.wc0*(z_bar*z_bar');

Xi = w_d.wj*zs*[xis -xis]';
H = zeros(q, length(P));
H(:, idxs) = (P_red\Xi')'; % Xi*P_red^{-1}

% compute covariance from measurement
% set sigma points w.r.t. noise
w_q = weights.q;
ys = w_q.sqrt_q_lambda * chol(R)';

% compute measurement sigma_points
zs = zeros(q, 2*q);
for j = 1:q
    y_j_p = y + ys(:, j);
    y_j_m = y - ys(:, j);
    z_j_p = z_aug(state, y_j_p);
    z_j_m =  z_aug(state, y_j_m);
    zs(:, j) = aug_phi_inv(aug_state, z_j_p);
    zs(:, q + j) = aug_phi_inv(aug_state, z_j_m);
end

% measurement mean
z_bar = w_q.wj * sum(zs, 2);

% prune mean before computing covariance
zs = zs - z_bar;
P_mes = w_q.wj * (zs*zs') + w_q.wc0*(z_bar*z_bar');

% compute augmented covariance
P_aug_aug = P_state + P_mes;
P_aug_prev = H*P;
aug_P = [P P_aug_prev';
    P_aug_prev P_aug_aug];
end