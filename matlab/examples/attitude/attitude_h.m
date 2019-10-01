function [y] = attitude_h(state)
%ATTITUDE_H Measurement
%
% Syntax: [y] = attitude_h(state)
%
% Inputs.
%    state - state
%
% Outputs:
%    y - measurement

% gravity for accelerometer measurement (m/s^2)
g = [0; 0; -9.81];
% normed magnetic fielf in Sweden for magnetometer measurement 
b = [0.33; 0; -0.95];

y = [state.Rot' * g;
    state.Rot' * b];
end