function [xi] = attitude_right_phi_inv(state, hat_state)
%ATTITUDE_RIGHT_PHI_INV inverse retraction
%
% Syntax: [xi] = attitude_right_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

xi = so3_log(state.Rot * hat_state.Rot');
end