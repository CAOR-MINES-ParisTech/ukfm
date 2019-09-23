# Benchmark

This folder contains different UKFs that have been compared and benchmarks with others Kalman filter based algorithms on the following vanilla problems.

## Content
This folder contains different UKFs that have been compared on the following vanilla problems.
- __localization__ : estimate the pose of a 2D robot equiped with velocity (wheel odometry) and position (GPS) measurements. 
- __attitude__: compute the 3D orientation  with an Inertial Measurement Unit (IMU) that disposes of gyro, accelerometers and magnetometers.
- __inertial navigation__: locate a vehicle which is navigating on flat Earth with inertial measurents and (known) landmark observations.
- __sphere__: estimate the state of a system living in a sphere.
- __2D SLAM robot__: estimate the pose of a 2D robot equiped with velocity (wheel odometry) along with the position of unknown landmarks, i.e. performs SLAM.
