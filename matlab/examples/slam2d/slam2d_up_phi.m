function [new_state] = slam2d_up_phi(state, xi)
%SLAM2D_UP_PHI retraction
%
% Syntax: [new_state] = slam2d_up_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = state.Rot * so2_exp(xi(1));
new_state.p = state.p + xi(2:3);
new_state.p_l = state.p_l + xi(4:5);
end
