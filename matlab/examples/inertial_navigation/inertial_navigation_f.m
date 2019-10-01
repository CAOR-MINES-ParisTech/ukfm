function [new_state] = inertial_navigation_f(state, omega, w, dt)
%INERTIAL_NAVIGATION_F Propagate state
%
% Syntax:  [new_state] = inertial_navigation_f(state, omega, w, dt)
%
% Inputs:
%    state - state
%    omega - input
%    w - input noise
%    dt - integration step
%
% Outputs:
%    new_state - propagated state

% gravity for accelerometer measurement (m/s^2)
g = [0; 0; -9.82];

acc = state.Rot * (omega.acc + w(4:6)) + g;

new_state.Rot = state.Rot * so3_exp((omega.gyro + w(1:3))*dt);
new_state.v = state.v + acc * dt;
new_state.p = state.p + state.v * dt + 1/2*acc*dt^2;
end

