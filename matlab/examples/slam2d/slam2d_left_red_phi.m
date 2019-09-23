function [new_state] = slam2d_left_red_phi(state, xi)
%SLAM2D_LEFT_RED_PHI retraction
%
% Syntax: [new_state] = slam2d_left_red_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state
new_state = slam2d_left_phi(state, xi);
end