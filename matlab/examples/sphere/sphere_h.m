function [y] = sphere_h(state)
%LOCALIZATION_H Measurement function
%
% Syntax: [y] = sphere_h(state)
%
% Inputs.
%    state - state
%
% Outputs:
%    y - measurement

y = state.p;
end

