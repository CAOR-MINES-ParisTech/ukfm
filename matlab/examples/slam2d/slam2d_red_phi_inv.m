function [xi] = slam2d_red_phi_inv(state, hat_state)
%SLAM2D_RED_PHI_INV inverse retraction
%
% Syntax: [xi] = slam2d_red_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

xi = [so2_log(state.Rot' * hat_state.Rot);
    hat_state.p - state.p];
end