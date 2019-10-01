function [new_state] = attitude_f(state, omega, w, dt)
%ATTITUDE_F Propagate state
%
% Syntax:  [new_state] = attitude_f(state, omega, w, dt)
%
% Inputs:
%    state - state
%    omega - input
%    w - input noise
%    dt - integration step
%
% Outputs:
%    new_state - propagated state

new_state.Rot = state.Rot * so3_exp((omega.gyro + w)*dt);
end

