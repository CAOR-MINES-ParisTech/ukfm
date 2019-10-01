function [up_state, up_P] = kf_update(state, P, H, r, R, phi)
%KF_UPDATE Kalman filter update
%
% Syntax: [up_state, up_P] = kf_update(state, P, y, h, R, phi)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    H - observability matrix
%    r - measurement residual
%    R - noise covariance matrix
%    phi - retraction, function
%
% Outputs:
%    up_state - updated state
%    up_P - updated covariance matrix

d = length(P);
S = H*P*H'+ R;

% Kalman gain
K = P*H'/S;

% update state
up_state = phi(state, K*r);

% update covariance
IKH = eye(d) - K*H;
up_P = IKH*P*IKH' + K*R*K';
end

