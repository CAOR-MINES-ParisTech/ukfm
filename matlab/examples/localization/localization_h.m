function [y] = localization_h(state)
%LOCALIZATION_H Measurement function
%
% Syntax: [y] = localization_h(state)
%
% Inputs.
%    state - state
%
% Outputs:
%    y - measurement

y = state.p;
end

