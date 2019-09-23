function [up_state, up_P] = slam2d_iekf_update(state, P, y, R, iekf_lmk)
%LOCALIZATION_IEKF_UPDATE Update step
%
% Syntax: [up_state, up_P] = slam2d_iekf_update(state, P, y, R, iekf_lmk)
%
% Inputs:
%    state - state
%    P - covariance matrix
%    y - measurement
%    R - noise covariance matrix
%    iekf_lmk - already observed landmark
%
% Outputs:
%    up_state - updated state
%    up_P - updated covariance matrix

N_y = length(find(y(3, :) > 0));
    
% set observalibity matrice and residual
H = zeros(0, length(P));
res = zeros(0);
tmp_state = state;
% update each landmark already in the filter
for i = 1:N_y
    idx = find(~(iekf_lmk - y(3, i)));
    if isempty(idx)
        continue
    end
    % indices of the robot and observed landmark in P
    H_i = zeros(2, length(P));
    H_i(:, 2:3) = -state.Rot';
    H_i(:, 2+(2*idx:2*idx+1)) = state.Rot';
    tmp_state.p_l = state.p_l(:, idx);
    % increase observabily matrix and residual
    res_i = y(1:2, i) - slam2d_h(tmp_state);
    H = [H; H_i];
    res = [res; res_i];
    R = blkdiag(R, R(1:2,1:2));
end

% update only if some landmards have been observed
if size(H, 1) > 0
    R = R(3:end, 3:end);
    % measurement uncertainty matrix
    S = H*P*H' + R;

    % gain matrix
    K = P*H' / S;

    % innovation
    xi = K * res;

    % update state
    up_state = slam2d_right_phi(state, xi);

    % update covariance
    IKH = (eye(length(P))-K*H);
    up_P = IKH*P*IKH' + K*R*K';
else
    up_state = state;
    up_P = P;
end
end