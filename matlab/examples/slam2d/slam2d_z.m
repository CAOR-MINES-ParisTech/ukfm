function [z] = slam2d_z(state, y)
%SLAM2D_Z Measurement function for augmenting state
%
% Syntax: [new_state] = slam2d_z(state, z)
%
% Inputs.
%    state - state
%    y - measurement
%
% Outputs:
%    new_state - state

z = state.Rot * y + state.p;
end

