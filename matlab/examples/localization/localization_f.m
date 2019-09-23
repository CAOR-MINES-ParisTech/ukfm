function [new_state] = localization_f(state, omega, w, dt)
%LOCALIZATION_F Propagate state
%
% Syntax:  [new_state] = localization_f(state, omega, w, dt)
%
% Inputs:
%    state - state
%    omega - input
%    w - input noise
%    dt - integration step
%
% Outputs:
%    new_state - propagated state

new_state.Rot = state.Rot * so2_exp((omega.gyro + w(3))*dt);
new_state.p = state.p + state.Rot*(omega.v + w(1:2))*dt;
end

