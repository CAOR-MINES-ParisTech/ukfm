# Examples
The implementation of UKF on parallelizable manifold on some examples.
## Get Started

If not already done, go at the Matlab prompt and execute `importukf`. You may save this path for your next Matlab sessions (via `savepath`). Start by the [2D robot localization tutorial](lien) to become familiar with UKF on parallelizable manifold.

## Content
This folder contains different UKFs that have been implemented on the following vanilla problems.
- __localization__ : estimate the pose of a 2D robot equiped with velocity (wheel odometry) and position (GPS) measurements. 
- __attitude__: compute the 3D orientation  with an Inertial Measurement Unit (IMU) that disposes of gyro, accelerometers and magnetometers.
- __inertial navigation__: locate a vehicle which is navigating on flat Earth with inertial measurents and (known) landmark observations.
- __sphere__: estimate the state of a system living in a sphere.
- __2D SLAM robot__: estimate the pose of a 2D robot equiped with velocity (wheel odometry) along with the position of unknown landmarks, i.e. performs SLAM.
-  __wifibot__ : the localization example with real data
- __IMU-GPS Kitti__ estimate the pose of a vehicle with IMU and GPS on real data from [1].

The complexity of the example is progressive. The __localization__ is a  descriptive tutorial for a vanilla UKF on a simple robot problem which is tested on real data in __wifibot__. __attitude__, __inertial navigation__ and __sphere__ are two applications of the method with intermediate difficulty. __2D SLAM robot__ include advanced tricks to perform efficient UKF steps when only a part of the state has non-linear dynamics, and __IMU-GPS Kitti__ designs an UKF  that is apply to real data from the car of the KITTI dataset [1].

We use numerical values for the above exemples in [2-6]. These papers can also be used to have detailled descriptions of the models.

[1] Geiger, A., Lenz, P., Stiller, C., & Urtasun, R. (2013). Vision meets robotics: The KITTI dataset. _The International Journal of Robotics Research_, _32_(11), 1231-1237.
[2] Barrau, A., & Bonnabel, S. (2016). The invariant extended Kalman filter as a stable observer. _IEEE Transactions on Automatic Control_, _62_(4), 1797-1812.
[3] Vasconcelos, J. F., Cunha, R., Silvestre, C., & Oliveira, P. (2010). A nonlinear position and attitude observer on SE (3) using landmark measurements. _Systems & Control Letters_, _59_(3-4), 155-166.
[4] Manon Kok, Jeroen D. Hol and Thomas B. Schön (2017), "Using Inertial Sensors for Position and Orientation Estimation", Foundations and Trends® in Signal Processing: Vol. 11: No. 1-2, pp 1-153.
[5] Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2013). A quadratic-complexity observability-constrained unscented Kalman filter for SLAM. _IEEE Transactions on Robotics_, _29_(5), 1226-1243.
[6] Brossard, M., Bonnabel, S., & Condomines, J. P. (2017, September). Unscented Kalman filtering on Lie groups. In _2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ (pp. 2485-2491). IEEE.
[7] https://github.com/borglab/gtsam/blob/develop/matlab/gtsam_examples/IMUKittiExampleGPS.m
[8] Svacha, J., Loianno, G., & Kumar, V. (2019). Inertial Yaw-Independent Velocity and Attitude Estimation for High-Speed Quadrotor Flight. _IEEE Robotics and Automation Letters_, _4_(2), 1109-1116.


