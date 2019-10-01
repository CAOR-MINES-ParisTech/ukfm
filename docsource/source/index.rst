Unscented Kalman Filtering on (Parallelizable) Manifolds
================================================================================
About
--------------------------------------------------------------------------------
**UKF-M**, for Unscented Kalman Filtering on (Parallelizable) Manifolds, is a
novel methodology for implementing unscented Kalman filters both  on manifolds
and Lie groups. Beyond *filtering performances*, the main advantages of the
approach are its *versatility*, as the method applies to numerous state
estimation problems, and its *simplicity of implementation* for practitioners
which are not necessarily familiar with manifolds and Lie groups.

This repo contains two independent Python and Matlab implementations - we
recommend Python - for quickly implementing and testing the approach. If you use
this project for your research, please cite:

.. highlight:: bibtex
.. code-block:: bibtex

    @article{brossard2019Code,
        author={Martin Brossard and Axel Barrau and Silvère Bonnabel},
        title={{A Code for Unscented Kalman Filtering on Manifolds (UKF-M)}},
        year={2019},
    }

.. toctree::
   :maxdepth: 4
   :hidden:
   
   install
   Tutorial <auto_examples/localization>
   examples
   benchmarks
   filter
   model
   geometry
   matlab
   license
   bibliography

Download
--------------------------------------------------------------------------------
The repo contains tutorials, documentation and can be downloaded from
https://github.com/CAOR-MINES-ParisTech/ukfm.

The paper *A Code for Unscented Kalman Filtering on Manifolds (UKF-M)*
related to this code is available at this `url
<https://cloud.mines-paristech.fr/index.php/s/uUjOhxaKp4v9yJT/download>`_. 

Getting Started
--------------------------------------------------------------------------------
1. Download the latest source code from `GitHub
<https://github.com/CAOR-MINES-ParisTech/ukfm>`_ (see :ref:`Installation
<installation>`).

2. Follow  the :ref:`Tutorial <localization_tutorial>` for an introduction to
the methodology. 

The rest of the documentation is build on the Python code. For the Matlab
user, see :ref:`here <matlab>`.

Examples
--------------------------------------------------------------------------------
Below is a list of examples from which the unscented Kalman filter on
parallelizable manifolds has been implemented:

-  2D robot localization (both for introduction on simulated data and on real
   data).

-  3D Attitude estimation with an Inertial Measurement Unit (IMU) equipped with
   gyros, accelerometers and magnetometers.

-  3D inertial navigation on flat Earth with observations of known landmarks.

-  2D Simultaneous Localization And Mapping (SLAM).

-  IMU-GNSS sensor-fusion for a vehicle on the KITTI dataset.

-  Spherical pendulum example, where the state lives on the 2-sphere manifold.

See more details in the :ref:`examples` Section.

Support
--------------------------------------------------------------------------------
Please, use the `GitHub issue tracker
<https://github.com/CAOR-MINES-ParisTech/ukfm>`_ for questions, bug reports,
feature requests/additions, etc.

Acknowledgments
--------------------------------------------------------------------------------
The library was written by `Martin Brossard
<mailto:martin.brossard@mines-paristech.fr>`_ ^, `Axel Barrau
<mailto:axel.barrau@safrangroup.com>`_ ^ and `Silvère Bonnabel
<mailto:silvere.bonnabel@mines-paristech.fr>`_ ^.

^ `MINES ParisTech <http://www.mines-paristech.eu/>`_ , PSL Research University,
Centre for Robotics, 60 Boulevard Saint-Michel, 75006 Paris, France.
