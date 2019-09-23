function [H, r] = ukf_jacobian_update(state, P, y, h, phi, weights, idxs)
%UKF_JACOBIAN_UPDATE Update the UKF state
%
% Syntax: [H, r] = ukf_jacobian_update(state, P, y, h, phi, weights, idxs)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement, vector
%    h - measurement function, function
%    phi - retraction, function
%    weights - weight parameters of UKF
%    idxs - index of the error state in P that are present in measurement
%
% Outputs:
%    H - Jacobian
%    r - residual

TOL = 1e-9; % tolerance for assuring positivness of P

P_red = P(idxs, idxs);
% set variable size
d = length(P_red);
l = length(y);

P_red = P_red + TOL*eye(d);

% set sigma points
w_u = weights.u;
xi_mat = w_u.sqrt_d_lambda * chol(P_red)';

% compute measurement sigma_points
y_mat = zeros(l, 2*d);
y_hat = h(state);
for j = 1:d
    chi_j_plus = phi(state, xi_mat(:, j));
    chi_j_minus = phi(state, -xi_mat(:, j));
    y_mat(:, j) = h(chi_j_plus);
    y_mat(:, d + j) = h(chi_j_minus);
end

% measurement mean
y_bar = w_u.wm0 * y_hat + w_u.wj * sum(y_mat, 2);

% prune mean before computing covariance
y_mat = y_mat - y_bar;

Y = w_u.wj*y_mat*[xi_mat -xi_mat]';
H_idx = (P_red\Y')'; % Y*P_red^{-1}

H = zeros(length(y), length(P));
H(:, idxs) = H_idx;

% compute residual
r = (y - y_bar);
end