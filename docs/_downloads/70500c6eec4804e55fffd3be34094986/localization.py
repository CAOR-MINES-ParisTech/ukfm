"""
********************************************************************************
2D Robot Localization - Benchmark
********************************************************************************

Goals of this script:

- implement different UKFs on the 2D robot localization example.

- design the Extended Kalman Filter (EKF) and the Invariant Extended
  Kalman Filter (IEKF) :cite:`barrauInvariant2017` for the given problem.

- compare the different algorithms with Monte-Carlo simulations.

*We assume the reader is already familiar with the considered problem described
in the tutorial.*

We previously designed an UKF with a standard uncertainty representation. An
advantage of the versatility of the UKF is to speed up implementation, tests,
and comparision UKF with different uncertainty representations. Indeed, for the
given problem, three different UKFs emerge, defined respectively as:

1) The state is embedded in :math:`SO(2) \\times \mathbb{R}^2`, as in the
   example, where:

   * the retraction :math:`\\varphi(.,.)` is the :math:`SO(2)` exponential map
     for orientation and the standard vector addition for robot position.

   * the inverse retraction :math:`\\varphi^{-1}(.,.)` is the :math:`SO(2)`
     logarithm for orientation and the standard vector subtraction for
     position.

2) The state is embedded in :math:`SE(2)` with left multiplication, i.e.

   - the retraction :math:`\\varphi(.,.)` is the :math:`SE(2)` exponential,
     where the state multiplies on the left the retraction
     :math:`\\boldsymbol{\\xi}`.

   - the inverse retraction :math:`\\varphi^{-1}(.,.)` is the :math:`SE(2)`
     logarithm.

   - this left UKF on :math:`SE(2)` corresponds to the Invariant Extended Kalman
     Filter (IEKF) recommended in :cite:`barrauInvariant2017`. We have
     theoretical reason to choose this retraction.

3) The state is embedded in :math:`SE(2)` with right multiplication, i.e.

   - the retraction :math:`\\varphi(.,.)` is the :math:`SE(2)` exponential,
     where the state multiplies on the right the retraction
     :math:`\\boldsymbol{\\xi}`.

   - the inverse retraction :math:`\\varphi^{-1}(.,.)` is the :math:`SE(2)`
     logarithm.

   - we do not dispose of reason to choose this retraction but test it as it is
     so simple.

We tests the different filters with the same noise parameter setting and on
simulation with strong initial heading error. We will see how perform the
filters compared to extended Kalman filters.
"""

################################################################################
# Import
# ==============================================================================
from ukfm import SO2, UKF, EKF
from ukfm import LOCALIZATION as MODEL
import ukfm
import numpy as np
import matplotlib
ukfm.utils.set_matplotlib_config()

################################################################################
# We compare the different filters on a large number of Monte-Carlo runs.

# Monte-Carlo runs
N_mc = 100

################################################################################
# Simulation Setting
# ==============================================================================
# We set the simulation as in :cite:`barrauInvariant2017`, section IV. The robot
# drives along a 10 m diameter circle for 40 seconds with high rate odometer
# measurements (100 Hz) and low rate GPS measurements (1 Hz). The vehicle gets
# moderate angular velocity uncertainty and highly precise linear velocity. The
# initial values of the heading error is very strong, **45° standard
# deviation**, while the initial position is known.

# sequence time (s)
T = 40
# odometry frequency (Hz)
odo_freq = 100
# create the model
model = MODEL(T, odo_freq)

# odometry noise standard deviation
odo_std = np.array([0.01,  # speed (v/m)
                    0.01,  # speed (v/m)
                    1 / 180 * np.pi])  # angular speed (rad/s)

# GPS frequency (Hz)
gps_freq = 1
# GPS noise standard deviation (m)
gps_std = 1
# radius of the circle trajectory (m)
radius = 5
# initial heading error standard deviation
theta0_std = 45 / 180 * np.pi

################################################################################
# Filter Design
# ==============================================================================
# Additionally to the three UKFs, we compare them to an Extended Kalman FIlter
# (EKF) and an Invariant EKF (IEKF). The EKF has the same uncertainty
# representation as the UKF with the retraction on :math:`SO(2) \times
# \mathbb{R}^2`, whereas the IEKF has the same uncertainty representation as the
# UKF with the left retraction on :math:`SE(2)`.

# propagation noise matrix
Q = np.diag(odo_std ** 2)

# measurement noise matrix
R = gps_std ** 2 * np.eye(2)

# initial covariance matrix
P0 = np.zeros((3, 3))
# we take into account initial heading error
P0[0, 0] = theta0_std ** 2

# sigma point parameter
alpha = np.array([1e-3, 1e-3, 1e-3])

################################################################################
# We set error variables before launching Monte-Carlo simulations. As we have
# five similar methods, the code is redundant.

ukf_err = np.zeros((N_mc, model.N, 3))
left_ukf_err = np.zeros_like(ukf_err)
right_ukf_err = np.zeros_like(ukf_err)
iekf_err = np.zeros_like(ukf_err)
ekf_err = np.zeros_like(ukf_err)

################################################################################
# We record Normalized Estimation Error Squared (NEES) for consistency
# evaluation (see Results).

ukf_nees = np.zeros((N_mc, model.N, 2))
left_ukf_nees = np.zeros_like(ukf_nees)
right_ukf_nees = np.zeros_like(ukf_nees)
iekf_nees = np.zeros_like(ukf_nees)
ekf_nees = np.zeros_like(ukf_nees)

################################################################################
# Monte-Carlo Runs
# ==============================================================================
# We run the Monte-Carlo through a for loop.
#
# .. note::
#
# We sample for each Monte-Carlo run an initial heading error from the true
# distribution in :math:`\mathbf{P}_0`. This is the correct manner to compare
# the filters but requires more  Monte-Carlo samples than a fixed value.
#

for n_mc in range(N_mc):
    print("Monte-Carlo iteration(s): " + str(n_mc + 1) + "/" + str(N_mc))
    # simulation true trajectory
    states, omegas = model.simu_f(odo_std, radius)
    # simulate measurement
    ys, one_hot_ys = model.simu_h(states, gps_freq, gps_std)

    # initialize filter with inaccurate state
    state0 = model.STATE(
        Rot=states[0].Rot.dot(SO2.exp(theta0_std * np.random.randn(1))),
        p=states[0].p
    )

    # define the filters
    ukf = UKF(state0=state0,
              P0=P0,
              f=model.f,
              h=model.h,
              Q=Q,
              R=R,
              phi=model.phi,
              phi_inv=model.phi_inv,
              alpha=alpha)

    left_ukf = UKF(state0=state0,
                   P0=P0,
                   f=model.f,
                   h=model.h,
                   Q=Q,
                   R=R,
                   phi=model.left_phi,
                   phi_inv=model.left_phi_inv,
                   alpha=alpha)

    right_ukf = UKF(state0=state0,
                    P0=P0,
                    f=model.f,
                    h=model.h,
                    Q=Q,
                    R=R,
                    phi=model.right_phi,
                    phi_inv=model.right_phi_inv,
                    alpha=alpha)

    iekf = EKF(model=model,
               state0=state0,
               P0=P0,  #  correct for your situation
               FG_ana=model.iekf_FG_ana,
               H_ana=model.iekf_H_ana,
               Q=Q,
               R=R,
               phi=model.left_phi)

    ekf = EKF(model=model,
              state0=state0,
              P0=P0,
              FG_ana=model.ekf_FG_ana,
              H_ana=model.ekf_H_ana,
              Q=Q,
              R=R,
              phi=model.phi)

    # variables for recording estimates of the Monte-Carlo run
    ukf_states = [state0]
    left_states = [state0]
    right_states = [state0]
    iekf_states = [state0]
    ekf_states = [state0]

    ukf_Ps = np.zeros((model.N, 3, 3))
    left_ukf_Ps = np.zeros_like(ukf_Ps)
    right_ukf_Ps = np.zeros_like(ukf_Ps)
    ekf_Ps = np.zeros_like(ukf_Ps)
    iekf_Ps = np.zeros_like(ukf_Ps)

    ukf_Ps[0] = P0
    left_ukf_Ps[0] = P0
    right_ukf_Ps[0] = P0
    ekf_Ps[0] = P0
    iekf_Ps[0] = P0

    # measurement iteration number
    k = 1

    # filtering loop
    for n in range(1, model.N):
        ukf.propagation(omegas[n-1], model.dt)
        left_ukf.propagation(omegas[n-1], model.dt)
        right_ukf.propagation(omegas[n-1], model.dt)
        iekf.propagation(omegas[n-1], model.dt)
        ekf.propagation(omegas[n-1], model.dt)
        # update only if a measurement is received
        if one_hot_ys[n] == 1:
            ukf.update(ys[k])
            left_ukf.update(ys[k])
            right_ukf.update(ys[k])
            iekf.update(ys[k])
            ekf.update(ys[k])
            k = k + 1

        ukf_states.append(ukf.state)
        left_states.append(left_ukf.state)
        right_states.append(right_ukf.state)
        iekf_states.append(iekf.state)
        ekf_states.append(ekf.state)

        ukf_Ps[n] = ukf.P
        left_ukf_Ps[n] = left_ukf.P
        right_ukf_Ps[n] = right_ukf.P
        iekf_Ps[n] = iekf.P
        ekf_Ps[n] = ekf.P

    # get state trajectory
    Rots, ps = model.get_states(states, model.N)
    ukf_Rots, ukf_ps = model.get_states(ukf_states, model.N)
    left_ukf_Rots, left_ukf_ps = model.get_states(left_states, model.N)
    right_ukf_Rots, right_ukf_ps = model.get_states(right_states, model.N)
    iekf_Rots, iekf_ps = model.get_states(iekf_states, model.N)
    ekf_Rots, ekf_ps = model.get_states(ekf_states, model.N)

    # record errors
    ukf_err[n_mc] = model.errors(Rots, ukf_Rots, ps, ukf_ps)
    left_ukf_err[n_mc] = model.errors(Rots, left_ukf_Rots, ps, left_ukf_ps)
    right_ukf_err[n_mc] = model.errors(Rots, right_ukf_Rots, ps, right_ukf_ps)
    iekf_err[n_mc] = model.errors(Rots, iekf_Rots, ps, iekf_ps)
    ekf_err[n_mc] = model.errors(Rots, ekf_Rots, ps, ekf_ps)

    # record NEES
    ukf_nees[n_mc] = model.nees(ukf_err[n_mc], ukf_Ps, ukf_Rots, ukf_ps, 'STD')
    left_ukf_nees[n_mc] = model.nees(left_ukf_err[n_mc], left_ukf_Ps,
                                     left_ukf_Rots, left_ukf_ps, 'LEFT')
    right_ukf_nees[n_mc] = model.nees(right_ukf_err[n_mc], right_ukf_Ps,
                                      right_ukf_Rots, right_ukf_ps, 'RIGHT')
    iekf_nees[n_mc] = model.nees(iekf_err[n_mc], iekf_Ps, iekf_Rots, iekf_ps,
                                 'LEFT')
    ekf_nees[n_mc] = model.nees(ekf_err[n_mc], ekf_Ps, ekf_Rots, ekf_ps, 'STD')

################################################################################
# Results
# ==============================================================================
# Once simulations are done, we compare the algorithms. Let us first visualize
# the results in term of estimated robot trajectory (for the last run) and the
# errors w.r.t. orientation and position (averaged over Monte-Carlo). As
# simulations have random process, the trajectory plot just gives us an
# indication but not a proof of filter performances.

ukf_e, left_ukf_e, right_ukf_e, iekf_e, ekf_e = model.benchmark_plot(
    ukf_err, left_ukf_err, right_ukf_err, iekf_err, ekf_err, ps, ukf_ps, 
    left_ukf_ps, right_ukf_ps, ekf_ps, iekf_ps)

################################################################################
# A statement seem obvious, we have two groups of filters: group 1) consists of
# EKF and :math:`SO(2) \times \mathbb{R}^2` UKF; and group 2) have IEKF, left
# :math:`SE(2)` UKF and right :math:`SE(2)` UKF (the curves of these filters are
# superposed). The second group is visibly highly better regarding position
# estimation.
#
# More statictical is to compute the results averaged over all the Monte-Carlo.
# Let us compute the Root Mean Squared Error (RMSE) for each method both for the
# orientation and the position.

model.benchmark_print(ukf_e, left_ukf_e, right_ukf_e, iekf_e, ekf_e)

################################################################################
# They confirm the the results on the plot.
#
# A consistency metric is the Normalized Estimation Error Squared (NEES).
# Classical criteria used to evaluate the performance of an estimation method,
# like the RMSE, do not inform about consistency as they do not take into
# account the uncertainty returned by the filter. This point is addressed by the
# NEES, which computes the average squared value of the error, normalized by the
# covariance matrix of the filter. The case NEES>1 reveals an inconsistency
# issue: the actual uncertainty is higher than the computed uncertainty.

model.nees_print(ukf_nees, left_ukf_nees, right_ukf_nees, iekf_nees, ekf_nees)

################################################################################
# As the filters are initialized with perfect position and zero covariance
# w.r.t. position, we compute NEES only after 20 s for avoiding numerical issues
# (during the first secondes of the trajectory the covariance matrix
# :math:`\mathbf{P}_n` is very low so inverting it leads to insignificantly high
# numbers). Results are clear, IEKF and :math:`SE(2)` are the more consistent.
# For the considered example, it seems that the UKFs are slightly less
# optimistic that their EKF counterparts.

################################################################################
# **Which filter is the best ?** We expected it is the left UKF as it takes
# advantage of both theory of IEKF uncertainty representation and better
# non-linearity noise incorporation of UKF compared to (I)EKF. Depending on
# which source of error (sensor noise, initial condition) is the most important,
# it can lead to different results. Indeed, in this setting, **left IEKF**,
# **right UKF** and **IEKF** filters obtain similar accurate results, that
# clearly outperform :math:`SO(2) \times \mathbb{R}^2` UKF, and one could expect
# encounter situations where the left UKF outperforms the right UKF.
#
# .. note::
#
#    We have set all the filters with the same "true" noise covariance 
#    parameters. However, both EKF and UKF based algorithms may better deal  ,
#    with non-linearity  by e.g. inflated propagation noise covariance.
#

################################################################################
# Conclusion
# ==============================================================================
# This script compares different algorithms for 2D robot localization. Two
# groups of filters emerge: the :math:`SO(2) \times \mathbb{R}^2` UKF and the
# EKF represent the first group; and the left :math:`SE(2)` UKF, the right
# :math:`SE(2)` UKF and the IEKF constitute the second group. For the considered
# set of parameters, it is evident that embedded the state in :math:`SE(2)` is
# advantageous for state estimation. Choosing then between left UKF, right UKF
# or IEKF has negligible effect (with the considered simulation setting).
#
# You can now:
#
# * compare the filters in different scenarios. Indeed, UKF and their (I)EKF
#   counterparts may obtain different results when noise is e.g. inflated or
#   with different initial conditions or trajectory.
#
# * testing the filters in a slightly different model (e.g. with orientation
#   measurement), which is straightforward for the UKFs.
#
# * address the problem of 3D attitude estimations, see the Examples section.
