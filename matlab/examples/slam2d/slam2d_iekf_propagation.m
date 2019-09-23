function [new_state, new_P] = slam2d_iekf_propagation(state, P, omega, ...
             dt, Q)
%SLAM2D_IEKF_PROPAGATION Propagation step
%
% Syntax: [new_state, new_P] = slam2d_iekf_propagation(state, ...
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
new_state = slam2d_f(state, omega, zeros(2, 1), dt);

% propagate covariance
J = [0 -1; 1 0];
G = zeros(length(P), 2);
G(2:3, 1) = state.Rot*[1; 0]*dt;
G(1, 2) = dt;
p_temp = -J*[state.p state.p_l];
G(2:end, 2) = p_temp(:)*dt;
new_P = P + G*Q*G';
end

