function [y] = pendulum_h(state)
%PENDULUM_H Measurement function
%
% Syntax: [y] = pendulum_h(state)
%
% Inputs.
%    state - state
%
% Outputs:
%    y - measurement

L = 1.3; % wire length (m)
e3 = -[0; 0; 1];
H = [[0, 1, 0]; [0, 0, 1]];
x = L*state.Rot*e3;
y = H*x;
end

