function [up_state, up_P] = ukf_update(state, P, y, h, phi, R, weights)
%UKF_UPDATE Update step of the UKF
%
% Syntax: [up_state, up_P] = ukf_update(state, P, y, h, phi, R, weights)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement
%    h - measurement function, function
%    phi - retraction, function
%    R -noise covariance matrix
%    weights - weight parameters
%
% Outputs:
%    up_state - updated state
%    up_P - updated covariance matrix

TOL = 1e-9; % tolerance for assuring positivness of P

% set variable size
d = length(P);
l = length(y);

P = P + TOL*eye(d);

% set sigma points
w_u = weights.u;
xis = w_u.sqrt_d_lambda * chol(P)';

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
y_hat = y_hat - y_bar;

% compute covariance and cross covariance matrices
P_yy = w_u.wc0*(y_hat*y_hat') + w_u.wj*(ys*ys');
P_yy = P_yy + R;
P_xiy = w_u.wj*[xis -xis]*ys';

% Kalman gain
K = P_xiy / P_yy;

% update state
xi_plus = K * (y - y_bar);
up_state = phi(state, xi_plus);

% update covariance
up_P = P - K * P_yy * K';
up_P = (up_P+up_P')/2;
end

