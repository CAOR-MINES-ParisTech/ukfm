function [xi] = attitude_phi_inv(state, hat_state)
%ATTITUDE_PHI_INV inverse retraction
%
% Syntax: [xi] = attitude_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

xi = so3_log(hat_state.Rot' * state.Rot);
end

