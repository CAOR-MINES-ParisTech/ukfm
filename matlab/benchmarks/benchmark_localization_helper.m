% get estimates and plot figure
state_table = struct2table(true_state);
Rot = state_table.Rot;
p = cell2mat(state_table.p')';

ukf_state_table = struct2table(ukf_states);
ukf_Rot = ukf_state_table.Rot;
ukf_p = cell2mat(ukf_state_table.p')';
ukf_err_p = vecnorm((p-ukf_p)');

ukf_left_state_table = struct2table(ukf_left_states);
ukf_left_Rot = ukf_left_state_table.Rot;
ukf_left_p = cell2mat(ukf_left_state_table.p')';
ukf_left_err_p = vecnorm((p-ukf_left_p)');

ukf_right_state_table = struct2table(ukf_right_states);
ukf_right_Rot = ukf_right_state_table.Rot;
ukf_right_p = cell2mat(ukf_right_state_table.p')';
ukf_right_err_p = vecnorm((p-ukf_right_p)');

ekf_state_table = struct2table(ekf_states);
ekf_Rot = ekf_state_table.Rot;
ekf_p = cell2mat(ekf_state_table.p')';
ekf_err_p = vecnorm((p-ekf_p)');

iekf_state_table = struct2table(iekf_states);
iekf_Rot = iekf_state_table.Rot;
iekf_p = cell2mat(iekf_state_table.p')';
iekf_err_p = vecnorm((p-iekf_p)');

% get orientation error
t = linspace(0, dt*N, N);
ukf_err_rot = zeros(N, 1);
ukf_left_err_rot = zeros(N, 1);
ukf_right_err_rot = zeros(N, 1);
ekf_err_rot = zeros(N, 1);
iekf_err_rot = zeros(N, 1);
for n = 1:N
    ukf_err_rot(n) = so2_log(ukf_Rot{n}' * Rot{n});
    ukf_left_err_rot(n) = so2_log(ukf_left_Rot{n}' * Rot{n});
    ukf_right_err_rot(n) = so2_log(ukf_right_Rot{n}' * Rot{n});
    ekf_err_rot(n) = so2_log(ekf_Rot{n}' * Rot{n});
    iekf_err_rot(n) = so2_log(iekf_Rot{n}' * Rot{n});
end

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');

% plot position
fig = figure;
hold on;
grid on;
title('Position in $xy$-plan for a Monte-Carlo run')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
plot(p(:, 1), p(:, 2), 'k');
plot(ukf_p(:, 1), ukf_p(:, 2), 'm');
plot(ukf_left_p(:, 1), ukf_left_p(:, 2), 'g');
plot(ukf_right_p(:, 1), ukf_right_p(:, 2), 'c');
plot(ekf_p(:, 1), ekf_p(:, 2), '');
plot(iekf_p(:, 1), iekf_p(:, 2), 'b');
axis equal;
legend('true position', 'standard UKF', 'left UKF', 'right UKF', 'EKF', 'IEKF [BB17]');
print(fig, 'benchmark/html/benchmark_localization_01', '-dpng', '-r600')

% plot attitude error
fig = figure;
title('Robot attitude error (deg)')
xlabel('$t$ (s)')
ylabel('orientation error (deg)')
hold on;
grid on;
% error
plot(t, 180/pi*ukf_err_rot, 'm');
plot(t, 180/pi*ukf_left_err_rot, 'g');
plot(t, 180/pi*ukf_right_err_rot, 'c');
plot(t, 180/pi*ekf_err_rot, '');
plot(t, 180/pi*iekf_err_rot, 'b');
legend('standard UKF', 'left UKF', 'right UKF', 'EKF', 'IEKF [BB17]');
print(fig, 'benchmark/html/benchmark_localization_02', '-dpng', '-r600')

% plot position error
fig = figure;
title('Robot position error (m)')
xlabel('$t$ (s)')
ylabel('error (m)')
hold on;
grid on;
% error
plot(t, ukf_err_p, 'm');
plot(t, ukf_left_err_p, 'g');
plot(t, ukf_right_err_p, 'c');
plot(t, ekf_err_p, '');
plot(t, iekf_err_p, 'b');
legend('standard UKF', 'left UKF', 'right UKF', 'EKF', 'IEKF [BB17]');
print(fig, 'benchmark/html/benchmark_localization_03', '-dpng', '-r600')

ukf_err_p = sprintf('%0.2f', sqrt(mean(ukf_err(2, :))/N));
ukf_left_err_p = sprintf('%0.2f', sqrt(mean(ukf_left_err(2, :))/N));
ukf_right_err_p = sprintf('%0.2f', sqrt(mean(ukf_right_err(2, :))/N));
ekf_err_p = sprintf('%0.2f', sqrt(mean(ekf_err(2, :))/N));
iekf_err_p = sprintf('%0.2f', sqrt(mean(iekf_err(2, :))/N));

ukf_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ukf_err(1, :))/N));
ukf_left_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ukf_left_err(1, :))/N));
ukf_right_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ukf_right_err(1, :))/N));
ekf_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ekf_err(1, :))/N));
iekf_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(iekf_err(1, :))/N));

disp(' ')
disp('Root Mean Square Error w.r.t. orientation (deg)');
disp("    -standard UKF: " + ukf_err_rot);
disp("    -left UKF    : " + ukf_left_err_rot);
disp("    -right UKF   : " + ukf_right_err_rot);
disp("    -EKF         : " + ekf_err_rot);
disp("    -IEKF        : " + iekf_err_rot);

disp(' ')
disp('Root Mean Square Error w.r.t. position (m)');
disp("    -standard UKF: " + ukf_err_p);
disp("    -left UKF    : " + ukf_left_err_p);
disp("    -right UKF   : " + ukf_right_err_p);
disp("    -EKF         : " + ekf_err_p);
disp("    -IEKF        : " + iekf_err_p);

