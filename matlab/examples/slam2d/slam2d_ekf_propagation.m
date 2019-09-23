function [new_state, new_P] = slam2d_ekf_propagation(state, P, omega, ...
             dt, Q)
%SLAM2D_EKF_PROPAGATION Propagation step
%
% Syntax: [new_state, new_P] = slam2d_ekf_propagation(state, ...
%             P, omega, dt, Q)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    omega - input
%    dt - integration step
%    Q - noise covariance matrix
%
% Outputs:
%    new_state - propagated state
%    new_P - propagated covariance matrix

% propagate mean state
new_state = slam2d_f(state, omega, zeros(3, 1), dt);

% propagate covariance
F = eye(length(P));
J = [0 -1;
     1  0];
F(2:3, 1) = state.Rot*J*[omega.v; 0]*dt;
G = zeros(length(P), 2);
G(2:3, 1) = state.Rot*[1; 0] * dt;
G(1, 2) = dt;

new_P = F*P*F' + G*Q*G';
end