.. _geometry:

Lie Groups
================================================================================
Implementation of the more used matrix Lie groups using numpy. The
implementation of :math:`SO(2)`, :math:`SE(2)`, :math:`SO(3)`, and :math:`SE(3)`
is based on the liegroups github `repo
<https://github.com/utiasSTARS/liegroups>`_. The implementation of
:math:`SE_k(2)` and :math:`SE_k(3)` works for any :math:`k>0`.

:math:`SO(2)`
--------------------------------------------------------------------------------
.. autoclass:: ukfm.SO2
    :members:

:math:`SE(2)`
--------------------------------------------------------------------------------
.. autoclass:: ukfm.SE2
    :members:

:math:`SE_k(2)`
--------------------------------------------------------------------------------
.. autoclass:: ukfm.SEK2
    :members:

:math:`SO(3)`
--------------------------------------------------------------------------------
.. autoclass:: ukfm.SO3
    :members:

:math:`SE(3)`
--------------------------------------------------------------------------------
.. autoclass:: ukfm.SE3
    :members:

:math:`SE_k(3)`
--------------------------------------------------------------------------------
.. autoclass:: ukfm.SEK3
    :members:

