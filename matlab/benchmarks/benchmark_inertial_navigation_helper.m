% get estimates and plot figure
state_table = struct2table(true_state);
p = cell2mat(state_table.p')';

ukf_state_table = struct2table(ukf_states);
ukf_p = cell2mat(ukf_state_table.p')';

ukf_left_state_table = struct2table(ukf_left_states);
ukf_left_p = cell2mat(ukf_left_state_table.p')';

ukf_right_state_table = struct2table(ukf_right_states);
ukf_right_p = cell2mat(ukf_right_state_table.p')';

ekf_state_table = struct2table(ekf_states);
ekf_p = cell2mat(ekf_state_table.p')';

iekf_state_table = struct2table(iekf_states);
iekf_p = cell2mat(iekf_state_table.p')';

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');

t = 0:dt:T-dt;

% plot position
fig = figure;
hold on;
grid on;
title('Position in $xy$-plan')
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
print(fig, 'benchmark/html/benchmark_inertial_navigation_01', '-dpng', '-r600')

% plot attitude error
% fig = figure;
% title('Robot attitude error')
% xlabel('$t$ (s)')
% ylabel('error (deg)')
% hold on;
% grid on;
% % error
% plot(t, 180/pi*ukf_err_rot, 'm');
% plot(t, 180/pi*ukf_left_err_rot, 'g');
% plot(t, 180/pi*ukf_right_err_rot, 'c');
% plot(t, 180/pi*ekf_err_rot, '');
% plot(t, 180/pi*iekf_err_rot, 'b');
% legend('standard UKF', 'left UKF', 'right UKF', 'EKF', 'IEKF [BB17]');
% print(fig, 'benchmark/html/benchmark_inertial_navigation_02', '-dpng', '-r600')

% position error
fig = figure;
hold on;
grid on;
title('Robot position error')
xlabel('$t$ (s)')
ylabel('error (m)')
t = 0:dt:T-dt;
plot(t, vecnorm((p-ukf_p)'), 'm');
plot(t, vecnorm((p-ukf_left_p)'), 'g');
plot(t, vecnorm((p-ukf_right_p)'), 'c');
plot(t, vecnorm((p-ekf_p)'), '');
plot(t, vecnorm((p-iekf_p)'), 'b');
legend('standard UKF', 'left UKF', 'right UKF', 'EKF', 'IEKF [BB17]');
print(fig, 'benchmark/html/benchmark_inertial_navigation_03', '-dpng', '-r600')



ukf_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ukf_err(1, :))/N));
ukf_left_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ukf_left_err(1, :))/N));
ukf_right_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ukf_right_err(1, :))/N));
ekf_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(ekf_err(1, :))/N));
iekf_err_rot = sprintf('%0.2f', 180/pi*sqrt(mean(iekf_err(1, :))/N));

ukf_err_p = sprintf('%0.2f', sqrt(mean(ukf_err(2, :))/N));
ukf_left_err_p = sprintf('%0.2f', sqrt(mean(ukf_left_err(2, :))/N));
ukf_right_err_p = sprintf('%0.2f', sqrt(mean(ukf_right_err(2, :))/N));
ekf_err_p = sprintf('%0.2f', sqrt(mean(ekf_err(2, :))/N));
iekf_err_p = sprintf('%0.2f', sqrt(mean(iekf_err(2, :))/N));

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