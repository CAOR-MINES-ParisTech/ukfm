function [new_state] = slam2d_left_aug_phi(state, xi)
%SLAM2D_LEFT_AUG_PHI retraction
%
% Syntax: [new_state] = slam2d_left_aug_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

chi = se2_exp(xi);
new_state.Rot = state.Rot * chi(1:2, 1:2);
new_state.p = state.p + state.Rot * chi(1:2, 3);
end