function [new_state] = slam2d_f(state, omega, w, dt)
%SLAM2D_F Propagate state for localization example
%
% Syntax:  [new_state] = slam2d_f(state, omega, w, dt)
%
% Inputs:
%    state - state
%    omega - input
%    w - input noise
%    dt - integration step
%
% Outputs:
%    new_state - propagated state

new_state.Rot = state.Rot * so2_exp((omega.gyro + w(2))*dt);
new_state.p = state.p + state.Rot*([omega.v + w(1); 0])*dt;
new_state.p_l = state.p_l;
end

