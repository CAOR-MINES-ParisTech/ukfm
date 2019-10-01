function [new_state] = slam2d_phi(state, xi)
%SLAM2D_PHI retraction
%
% Syntax: [new_state] = slam2d_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = state.Rot * so2_exp(xi(1));
new_state.p = state.p + xi(2:3);
l = (length(xi)-3)/2;
new_state.p_l = state.p_l + reshape(xi(4:end), 2, l);
end
