Filters
================================================================================

This page describes the base class for designed an UKF (:meth:`~ukfm.UKF`) and a
Jacobian UKF, :meth:`~ukfm.JUKF`, which is well adapted when the dimension of
the state is important. :meth:`~ukfm.JUKF` infers numerical Jacobian, is
relatively less intuitive and gets exactly the same results as
:meth:`~ukfm.UKF`. We finally add a base class for an Extended Kalman Filter
(:meth:`~ukfm.EKF`) that requires analytical Jacobian computation.

UKF
--------------------------------------------------------------------------------
.. autoclass:: ukfm.UKF
    :members:

JUKF
--------------------------------------------------------------------------------
.. autoclass:: ukfm.JUKF
    :members:

EKF
--------------------------------------------------------------------------------
.. autoclass:: ukfm.EKF
    :members:

