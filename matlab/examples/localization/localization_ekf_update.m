function [up_state, up_P] = localization_ekf_update(state, P, y, R)
%LOCALIZATION_EKF_UPDATE Update step
%
% Syntax: [up_state, up_P] = localization_ekf_update(state, P, y, R)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement
%    R - noise covariance matrix
%
% Outputs:
%    up_state - updated state
%    up_P - updated covariance matrix

% Observability matrix
H = [zeros(2, 1) eye(2)];

% measurement uncertainty matrix
S = H*P*H' + R;

% gain matrix
K = P*H' / S;

% innovation
xi = K * (y-localization_h(state));

% update state
up_state = localization_phi(state, xi);

% update covariance
up_P = (eye(3)-K*H)*P;
end

