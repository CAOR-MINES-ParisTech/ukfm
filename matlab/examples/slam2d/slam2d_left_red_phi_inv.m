function [xi] = slam2d_left_red_phi_inv(state, hat_state)
%SLAM2D_LEFT_RED_PHI_INV inverse retraction
%
% Syntax: [xi] = slam2d_left_red_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

l = size(state.p_l, 2);
chi = [state.Rot state.p state.p_l;
       zeros(l+1, 2) eye(l+1)];
hat_chi = [hat_state.Rot hat_state.p hat_state.p_l;
       zeros(l+1, 2) eye(l+1)];   

xi = se_k_2_log(se_k_2_inv(chi) * hat_chi);
end