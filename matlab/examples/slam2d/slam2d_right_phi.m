function [new_state] = slam2d_right_phi(state, xi)
%SLAM2D_RIGHT_PHI retraction
%
% Syntax: [new_state] = slam2d_right_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

chi = se_k_2_exp(xi);
new_state.Rot = chi(1:2, 1:2) * state.Rot;
new_state.p = chi(1:2, 3) + chi(1:2, 1:2) * state.p;
if length(chi) >= 4
    new_state.p_l =  chi(1:2, 4:end) + chi(1:2, 1:2) * state.p_l;
else
    new_state.p_l = state.p_l;
end
