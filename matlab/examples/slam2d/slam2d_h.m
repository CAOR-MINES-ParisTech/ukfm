function [y] = slam2d_h(state)
%SLAM2D_H Measurement function
%
% Syntax: [y] = slam2d_h(state)
%
% Inputs.
%    state - state
%
% Outputs:
%    y - measurement

y = state.Rot' * (state.p_l - state.p);
end

