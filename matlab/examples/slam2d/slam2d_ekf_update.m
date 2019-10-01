function [up_state, up_P] = slam2d_ekf_update(state, P, y, R, ekf_lmk)
%SLAM2D_EKF_UPDATE Update step
%
% Syntax: [up_state, up_P] = slam2d_ekf_update(state, P, y, R, ekf_lmk)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement
%    R - noise covariance matrix
%    ekf_lmk - already observed landmark
%
% Outputs:
%    up_state - updated state
%    up_P - updated covariance matrix

N_y = length(find(y(3, :) > 0));
    
% set observalibity matrice and residual
H = zeros(0, length(P));
res = zeros(0);

% set ukf state for update
tmp_state.Rot = state.Rot;
tmp_state.p = state.p;
J = [0 -1; 1 0];
% update each landmark already in the filter
for i = 1:N_y
    idx = find(~(ekf_lmk - y(3, i)));
    if isempty(idx)
        continue
    end
    % indices of the robot and observed landmark in P
    tmp_state.p_l = state.p_l(:, idx);
    H_i = zeros(2, length(P));
    H_i(:, 1) = -state.Rot'*J*(tmp_state.p_l-tmp_state.p);
    H_i(:, 2:3) = -state.Rot';
    H_i(:,2+(2*idx:2*idx+1)) = state.Rot';
    % increase observabily matrix and residual
    res_i = y(1:2, i) - slam2d_h(tmp_state);
    H = [H; H_i];
    res = [res; res_i];
    R = blkdiag(R, R(1:2,1:2));
end

% update only if some landmards have been observed
if size(H, 1) > 0

    % measurement uncertainty matrix
    S = H*P*H' + R(3:end, 3:end);

    % gain matrix
    K = P*H' / S;

    % innovation
    xi = K * res;

    % update state
    up_state = slam2d_phi(state, xi);
    % update covariance
    up_P = (eye(length(P))-K*H)*P;
else
    up_state = state;
    up_P = P;
end
end