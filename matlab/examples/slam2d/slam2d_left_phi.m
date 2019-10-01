function [new_state] = slam2d_left_phi(state, xi)
%SLAM2D_LEFT_PHI retraction
%
% Syntax: [new_state] = slam2d_left_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

chi = se_k_2_exp(xi);
new_state.Rot = state.Rot * chi(1:2, 1:2);
new_state.p = state.p + state.Rot * chi(1:2, 3);
if length(chi) >= 4
    new_state.p_l = state.p_l + new_state.Rot * chi(1:2, 4:end);
else
    new_state.p_l = state.p_l;
end