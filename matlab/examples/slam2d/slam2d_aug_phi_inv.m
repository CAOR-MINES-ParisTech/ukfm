function [xi] = slam2d_aug_phi_inv(state, aug_state)
%SLAM2D_AUG_PHI_INV inverse retraction
%
% Syntax: [xi] = slam2d_phi_inv(state, aug_state)
%
% Inputs:
%    state - state
%    aug_state - augmented state
%
% Outputs:
%    xi - uncertainty

xi = aug_state.p_l - state.p_l;
end