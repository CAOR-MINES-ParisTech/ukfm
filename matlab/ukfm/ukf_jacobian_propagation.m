function [new_state, new_P] = ukf_jacobian_propagation(state, P, ...
    omega, f, dt, phi, phi_inv, cholQ, weights, idxs)
%UKF_JACOBIAN_PROPAGATION Propagate the UKF state in the situation where a
% part of the state is stationnary or parameters
%
% Syntax: [new_state, new_P] = ukf_propagation(state, P, omega, f, dt, ...
%     phi, phi_inv, cholQ, weights, idxs)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    omega - input
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

% mean update
w = zeros(q, 1);
new_state = f(state, omega, w, dt);

% compute covariance w.r.t. state uncertainty
w_d = weights.d;

% set sigma points
xis = w_d.sqrt_d_lambda * chol(P_red)';
xis_new = zeros(d, 2*d);

% rectract sigma points onto manifold
for j = 1:d
    s_j_p = phi(state, xis(:, j));
    s_j_m = phi(state, -xis(:, j));
    s_j_p_new = f(s_j_p, omega, w, dt);
    s_j_m_new = f(s_j_m, omega, w, dt);
    xis_new(:, j) = phi_inv(new_state, s_j_p_new);
    xis_new(:, d + j) = phi_inv(new_state, s_j_m_new);
end

% compute covariance
xi_mean = w_d.wj * sum(xis_new, 2);
xis_new = xis_new - xi_mean;
Xi = w_d.wj * (xis_new*[xis -xis]');
F(idxs, idxs) = (P_red\Xi')'; % Xi*P_red^{-1}
new_P = F*P*F';

% compute covariance w.r.t. noise
w_q = weights.q;
xis_new = zeros(d, 2*q);

% rectract sigma points onto manifold
for j = 1:q
    w_p = w_q.sqrt_q_lambda * cholQ(:, j);
    w_m = -w_q.sqrt_q_lambda * cholQ(:, j);
    s_j_p_new = f(state, omega, w_p, dt);
    s_j_m_new = f(state, omega, w_m, dt);
    xis_new(:, j) = phi_inv(new_state, s_j_p_new);
    xis_new(:, q + j) = phi_inv(new_state, s_j_m_new);
end

% compute covariance
xi_mean = w_q.wj * sum(xis_new, 2);
xis_new = xis_new - xi_mean;
Q = w_q.wj * (xis_new*xis_new') + w_q.wc0*(xi_mean*xi_mean');

% sum covariances
new_P(idxs, idxs) = new_P(idxs, idxs) + Q;
end