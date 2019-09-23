.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_pendulum.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_pendulum.py:


********************************************************************************
Pendulum Example
********************************************************************************

The set of all points in the Euclidean space :math:`\mathbb{R}^{3}`, that lie on
the surface of the unit ball about the origin belong to the two-sphere manifold,
:math:`S(2)`, which is a two-dimensional manifold. Many mechanical systems such
as a spherical pendulum, double pendulum, quadrotor with a cable-suspended load,
evolve on either :math:`S(2)` or products comprising of :math:`S(2)`.

In this script, we estimate the state of a system living on the sphere but where
observations are standard vectors. You can have a text description of the 
spherical pendulum dynamics in :cite:`sjobergAn2019`, Section 7, and
:cite:`kotaruVariation2019`.

Import
==============================================================================


.. code-block:: default

    from scipy.linalg import block_diag
    import ukfm
    import numpy as np
    import matplotlib
    ukfm.utils.set_matplotlib_config()







Model and Simulation
==============================================================================
This script uses the ``INERTIAL_NAVIGATION`` model class that requires  the
sequence time and the model frequency to create an instance of the model.


.. code-block:: default


    # sequence time (s)
    T = 20
    # IMU frequency (Hz)
    model_freq = 100
    # create the model
    model = ukfm.PENDULUM(T, model_freq)







The true trajectory is computed along with empty inputs (the model does not
require input) after we define the noise standard deviation affecting the
dynamic.


.. code-block:: default


    # model standard-deviation noise (noise is isotropic)
    model_std = np.array([2/180*np.pi,  # orientation (rad/s)
                        0.5]) # orientation velocity (rad/s^2)

    # simulate true trajectory and noised input
    states, omegas = model.simu_f(model_std)







The state and the input contain the following variables:

.. highlight:: python
.. code-block:: python

  states[n].Rot  # 3d orientation (matrix)
  states[n].u    # 3d angular velocity
  omegas[n]      # empty input

We compute noisy measurements at low frequency based on the true state
sequence.


.. code-block:: default


    # observation noise standard deviation (m)
    obs_std = 0.1
    # simulate landmark measurements
    ys = model.simu_h(states, obs_std)







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
    Q = block_diag(model_std[0]**2*np.eye(3), model_std[1]**2*np.eye(3))
    # measurement noise matrix
    R = obs_std**2*np.eye(2)
    # initial error matrix
    P0 = np.zeros((6, 6))  # The state is perfectly initialized
    # sigma point parameters
    alpha = np.array([1e-3, 1e-3, 1e-3])







We initialize the filter with the true state.


.. code-block:: default


    state0 = model.STATE(
        Rot=states[0].Rot,
        u=states[0].u,
    )

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
    ukf_Ps = np.zeros((model.N, 6, 6))
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
error
model.plot_results(ukf_states, ukf_Ps, states)

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

   **Total running time of the script:** ( 0 minutes  12.505 seconds)


.. _sphx_glr_download_auto_examples_pendulum.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: pendulum.py <pendulum.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: pendulum.ipynb <pendulum.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
