.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_attitude.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_attitude.py:


********************************************************************************
Attitude Estimation with an IMU - Example
********************************************************************************

Goal of this script:

- applying the UKF for estimating 3D attitude from an IMU.

*We assume the reader is already familiar with the tutorial.*

Attitude estimation with an Inertial Measurement Unit (IMU). The filter fuses
measurements coming from gyro, accelerometer and magnetometer. The IMU does not
have any bias. We reproduce the simulation based on :cite:`kokUsing2017`.

Import
==============================================================================


.. code-block:: default

    from scipy.linalg import block_diag
    from ukfm import ATTITUDE as MODEL
    import ukfm
    import numpy as np
    import matplotlib
    ukfm.utils.set_matplotlib_config()







Model and Simulation
==============================================================================
This script uses the ``ATTITUDE`` model class that requires  the sequence time
and the IMU frequency to create an instance of the model.


.. code-block:: default


    # sequence time (s)
    T = 100
    # IMU frequency (Hz)
    imu_freq = 100
    # create the model
    model = MODEL(T, imu_freq)







The true trajectory is computed along with noisy inputs after we define the
noise standard deviation affecting the (accurate) IMU, where the platform is 2
s stationary and then have constant angular velocity around gravity.


.. code-block:: default


    # IMU standard-deviation noise (noise is isotropic)
    imu_std = np.array([5/180*np.pi,  # gyro (rad/s)
                        0.4,          # accelerometer (m/s**2)
                        0.2])         # magnetometer

    # simulate true trajectory and noisy inputs
    states, omegas = model.simu_f(imu_std)







The state and the input contain the following variables:

.. highlight:: python
.. code-block:: python

      states[n].Rot      # 3d orientation (matrix)
      omegas[n].gyro     # robot angular velocities

We compute noisy measurements based on the true state sequence.


.. code-block:: default


    ys = model.simu_h(states, imu_std)







A measurement ``ys[k]`` contains accelerometer and magnetometer measurements.

Filter Design and Initialization
------------------------------------------------------------------------------
We choose in this example to embed the state in :math:`SO(3)` with left
multiplication, such that:

- the retraction :math:`\varphi(.,.)` is the :math:`SO(3)` exponential map for
  orientation where the state multiplies the uncertainty on the left.

- the inverse retraction :math:`\varphi^{-1}(.,.)` is the :math:`SO(3)`
  logarithm for orientation.


.. code-block:: default


    # propagation noise matrix
    Q = imu_std[0]**2*np.eye(3)
    # measurement noise matrix
    R = block_diag(imu_std[1]**2*np.eye(3), imu_std[2]**2*np.eye(3))
    # initial error matrix
    P0 = np.zeros((3, 3))  # The state is perfectly initialized
    # sigma point parameters
    alpha = np.array([1e-3, 1e-3, 1e-3])







We initialize the filter with the true state.


.. code-block:: default


    state0 = model.STATE(Rot=states[0].Rot)

    ukf = ukfm.UKF(state0=state0,
                   P0=P0,
                   f=model.f,
                   h=model.h,
                   Q=Q,
                   R=R,
                   phi=model.phi,
                   phi_inv=model.phi_inv,
                   alpha=alpha)

    # set variables for recording estimates along the full trajectory
    ukf_states = [state0]
    ukf_Ps = np.zeros((model.N, 3, 3))
    ukf_Ps[0] = P0







Filtering
==============================================================================
The UKF proceeds as a standard Kalman filter with a simple for loop.


.. code-block:: default

    for n in range(1, model.N):
        # propagation
        ukf.propagation(omegas[n-1], model.dt)
        # update
        ukf.update(ys[n])
        # save estimates
        ukf_states.append(ukf.state)
        ukf_Ps[n] = ukf.P







Results
------------------------------------------------------------------------------
We plot the orientation as function of time along with the orientation
error.


.. code-block:: default

    model.plot_results(ukf_states, ukf_Ps, states, omegas)




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_attitude_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_attitude_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_attitude_003.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_attitude_004.png
            :class: sphx-glr-multi-img




We see the true trajectory starts by a small stationary step following
by constantly turning around the gravity vector (only the yaw is
increasing). As yaw is not observable with an accelerometer only, it is
expected that yaw error would be stronger than roll or pitch errors.

As UKF estimates the covariance of the error, we have plotted the 95%
confident interval (:math:`3\sigma`). We expect the error keeps behind this
interval, and in this situation the filter covariance output matches
especially well the error.

A cruel aspect of these curves is the absence of comparision. Is the filter
good ? It would be nice to compare it, e.g., to an extended Kalman filter.

Conclusion
==============================================================================
We have seen in this script how well works the UKF on parallelizable
manifolds for estimating orientation from an IMU.

You can now:

- address the UKF for the same problem with different noise parameters.

- add outliers in acceleration or magnetometer measurements.

- benchmark the UKF with different function errors and compare it to the
  extended Kalman filter in the Benchmarks section.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  20.857 seconds)


.. _sphx_glr_download_auto_examples_attitude.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: attitude.py <attitude.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: attitude.ipynb <attitude.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
