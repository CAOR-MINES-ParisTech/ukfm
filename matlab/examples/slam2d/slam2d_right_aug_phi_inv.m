function [xi] = slam2d_right_aug_phi_inv(state, aug_state)
%SLAM2D_RIGHT_AUG_PHI_INV inverse retraction
%
% Syntax: [xi] = slam2d_right_aug_phi_inv(state, aug_state)
%
% Inputs:
%    state - state
%    aug_state - augmented state
%
% Outputs:
%    xi - uncertainty

chi = [state.Rot state.p_l;
       zeros(1, 2) 1];
aug_chi = [aug_state.Rot aug_state.p_l;
       zeros(1, 2) 1];   
xi = se2_log(aug_chi * se2_inv(chi));
xi = xi(2:3);
end