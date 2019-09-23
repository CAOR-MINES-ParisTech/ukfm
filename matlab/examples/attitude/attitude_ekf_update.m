function [up_state, up_P] = attitude_ekf_update(state, P, y, R)
%ATTITUDE_EKF_UPDATE Update step
%
% Syntax: [up_state, up_P] = attitude_ekf_update(state, P, y, R)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement, vector
%    R -  noise covariance matrix
%
% Outputs:
%    up_state - updated state
%    up_P - updated covariance matrix

% gravity for accelerometer measurement (m/s^2)
g = [0; 0; -9.81];
% normed magnetic fielf in Sweden for magnetometer measurement 
b = [0.33; 0; -0.95];
% Observability matrix
H = zeros(6, 3);
H(1:3, :) = -state.Rot'*so3_wedge(g);
H(4:6, :) = -state.Rot'*so3_wedge(b);

% measurement uncertainty matrix
S = H*P*H' + R;

% gain matrix
K = P*H' / S;

% innovation
xi = K * (y-attitude_h(state));

% update state
up_state = attitude_right_phi(state, xi);

% update covariance
up_P = (eye(3)-K*H)*P;
end
