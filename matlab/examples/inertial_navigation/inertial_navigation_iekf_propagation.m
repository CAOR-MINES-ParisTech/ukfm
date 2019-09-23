function [new_state, new_P] = inertial_navigation_iekf_propagation(...
    state, P, omega, dt, Q)
%INERTIAL_NAVIGATION_IEKF_PROPAGATION Propagation step
%
% Syntax: [new_state, new_P] = inertial_navigation_iekf_propagation(...
%             state, P, omega, dt, Q)
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

% gravity for accelerometer measurement (m/s^2)
g = [0; 0; -9.82];

% propagate mean state
new_state = inertial_navigation_f(state, omega, zeros(6, 1), dt);

% propagate covariance
F = eye(9);
F(4:6, 1:3) = so3_wedge(g)*dt;
F(7:9, 1:3) = 1/2*so3_wedge(g)*dt^2;
F(7:9, 4:6) = dt*eye(3);

G = zeros(9, 6);
G(1:3, 1:3) = state.Rot*dt;
G(4:6, 4:6) = state.Rot*dt;
G(4:6, 1:3) = so3_wedge(state.v) * state.Rot*dt;
G(7:9, 1:3) = so3_wedge(state.p) * state.Rot*dt;

new_P = F*P*F' + G*Q*G';
end

