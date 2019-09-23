function [new_state, new_P] = attitude_ekf_propagation(state, P, omega, ...
             dt, Q)
%ATTITUDE_EKF_PROPAGATION Propagation step
%
% Syntax: [new_state, new_P] = attitude_ekf_propagation(state, P, omega, ...
%             dt, Q)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    omega - input, structure
%    dt - integration step
%    Q - noise covariance matrix
%
% Outputs:
%    new_state - propagated state
%    new_P - propagated covariance matrix

% propagate mean state
new_state = attitude_f(state, omega, zeros(3, 1), dt);

% propagate covariance
G = zeros(3, 3);
G(1:3, 1:3) = dt*state.Rot;
new_P = P + G*Q*G';
end