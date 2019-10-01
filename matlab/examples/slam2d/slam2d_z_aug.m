function [new_state] = slam2d_z_aug(state, y)
%SLAM2D_Z_AUG Measurement function for augmenting state
%
% Syntax: [new_state] = slam2d_z_aug(state, z)
%
% Inputs.
%    state - state
%    y - measurement
%
% Outputs:
%    new_state - state

new_state = state;
new_state.p_l = state.Rot * y + state.p;
end

