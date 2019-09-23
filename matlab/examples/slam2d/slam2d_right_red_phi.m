function [new_state] = slam2d_right_red_phi(state, xi)
%SLAM2D_RIGHT_RED_PHI retraction
%
% Syntax: [new_state] = slam2d_right_red_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state = slam2d_right_phi(state, xi);
end