Models
================================================================================
This page describes the models considered in the examples. Each class contains
propagation and measurement functions, different choices of retractions and
their inverse retractions. You can also obtain in the source code useful
functions specifically related to the problem, e.g. Jacobian for EKF, and helper
functions.

2D Robot Localization
--------------------------------------------------------------------------------
.. autoclass:: ukfm.LOCALIZATION
    :members:

3D Attitude Estimation with an IMU
--------------------------------------------------------------------------------
.. autoclass:: ukfm.ATTITUDE
    :members:

3D Inertial Navigation on Flat Earth
--------------------------------------------------------------------------------
.. autoclass:: ukfm.INERTIAL_NAVIGATION
    :members:

2D Robot SLAM
--------------------------------------------------------------------------------
.. autoclass:: ukfm.SLAM2D
    :members:

IMU-GNSS Fusion on the KITTI Dataset
--------------------------------------------------------------------------------
.. autoclass:: ukfm.IMUGNSS
    :members:

Spherical Pendulum
--------------------------------------------------------------------------------
.. autoclass:: ukfm.PENDULUM
    :members:
