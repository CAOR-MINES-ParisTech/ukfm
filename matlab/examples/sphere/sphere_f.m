function [new_state] = sphere_f(state, omega, w, dt)
%SPHERE_F Propagate state
%
% Syntax:  [new_state] = sphere_f(state, omega, w, dt)
%
% Inputs:
%    state - state
%    omega - input
%    w - input noise
%    dt - time between propagation
%
% Outputs:
%    new_state - propagated state

new_state.Rot = state.Rot * so2_exp((omega.gyro + w(3))*dt);

end

