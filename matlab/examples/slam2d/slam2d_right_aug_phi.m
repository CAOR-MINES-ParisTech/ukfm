function [new_state] = slam2d_right_aug_phi(state, xi)
%SLAM2D_RIGHT_AUG_PHI retraction
%
% Syntax: [new_state] = slam2d_right_aug_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

chi = se2_exp(xi);
new_state.Rot = chi(1:2, 1:2) * state.Rot;
new_state.p = chi(1:2, 3) + chi(1:2, 1:2) * state.p;
end