function [up_state, up_P] = inertial_navigation_iekf_update(state, P, y, R)
%INERTIAL_NAVIGATION_IEKF_UPDATE Update step of the IEKF
%
% Syntax: [up_state, up_P] = inertial_navigation_iekf_update(state, ...
%       P, y, R)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement
%    R -noise covariance matrix
%
% Outputs:
%    up_state - updated state
%    up_P - updated covariance matrix

% landmarks
N_ldk = 3;
ldk = [[0; 2; 2], [-2; -2; -2], [2; -2; -2]];
% Observability matrix
H = zeros(3*N_ldk, 9);
for n_ldk = 1:N_ldk
    H(3*n_ldk-2: 3*n_ldk, 1:3) = state.Rot'*so3_wedge(ldk(:, n_ldk));
    H(3*n_ldk-2: 3*n_ldk, 7:9) = -state.Rot';
end

% measurement uncertainty matrix
S = H*P*H' + R;

% gain matrix
K = P*H' / S;

% innovation
xi = K * (y-inertial_navigation_h(state));

% update state
up_state = inertial_navigation_right_phi(state, xi);

% update covariance
up_P = (eye(9)-K*H)*P;
end

