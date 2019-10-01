function [H, r] = ukf_jacobian_update(state, P, y, h, phi, weights, idxs)
%UKF_JACOBIAN_UPDATE Update the UKF state
%
% Syntax: [H, r] = ukf_jacobian_update(state, P, y, h, phi, weights, idxs)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement
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
xis = w_u.sqrt_d_lambda * chol(P_red)';

% compute measurement sigma_points
ys = zeros(l, 2*d);
y_hat = h(state);
for j = 1:d
    chi_j_plus = phi(state, xis(:, j));
    chi_j_minus = phi(state, -xis(:, j));
    ys(:, j) = h(chi_j_plus);
    ys(:, d + j) = h(chi_j_minus);
end

% measurement mean
y_bar = w_u.wm0 * y_hat + w_u.wj * sum(ys, 2);

% prune mean before computing covariance
ys = ys - y_bar;

Y = w_u.wj*ys*[xis -xis]';
H_idx = (P_red\Y')'; % Y*P_red^{-1}

H = zeros(length(y), length(P));
H(:, idxs) = H_idx;

% compute residual
r = (y - y_bar);
end