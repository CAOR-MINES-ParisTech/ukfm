function [new_state, new_P] = localization_iekf_propagation(state, ...
    P, omega, dt, Q)
%LOCALIZATION_IEKF_PROPAGATION Propagation step
%
% Syntax: [new_state, new_P] = localization_iekf_propagation(state, ...
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
new_state = localization_f(state, omega, zeros(3, 1), dt);

% propagate covariance
F = eye(3);
J = [0 -1;
     1  0];
F(2:3, 1) = J*omega.v*dt;
F(2, 3) = omega.gyro*dt;
F(3, 2) = -omega.gyro*dt;
G = dt*eye(3);

new_P = F*P*F' + G*Q*G';
end

