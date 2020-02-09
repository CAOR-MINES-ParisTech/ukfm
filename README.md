Unscented Kalman Filtering on (Parallelizable) Manifolds
================================================================================

About
--------------------------------------------------------------------------------
**UKF-M**, for Unscented Kalman Filtering on (parallelizable) Manifolds, is a
novel methodology for implementing unscented Kalman filter both  on manifold and
Lie groups. Beyond *filtering performances*, the main interests of the approach
are its *versatility*, as the method applies to numerous state estimation
problems, and its *simplicity of implementation* for practitioners not being
necessarily familiar with manifolds and Lie groups.

This repo contains two independent Python and Matlab implementations - we
recommend Python - for quickly implementing and testing the approach. If you use
this project for your research, please please cite:

    @inproceedings{brossard2020Code,
        author={Martin Brossard and Axel Barrau and Silvère Bonnabel},
        title={{A Code for Unscented Kalman Filtering on Manifolds (UKF-M)}},
        booktitle={2020 International Conference on Robotics and Automation (ICRA)},
        year={2020},
        organization={IEEE}
    }


Documentation
--------------------------------------------------------------------------------

The documentation is available at [https://caor-mines-paristech.github.io/ukfm/](https://caor-mines-paristech.github.io/ukfm/).

The paper *A Code for Unscented Kalman Filtering on Manifolds (UKF-M)* related
to this code is available at  this [url](https://cloud.mines-paristech.fr/index.php/s/uUjOhxaKp4v9yJT/download).

Download
--------------------------------------------------------------------------------
The repo contains tutorials, documentation and that can be downloaded from
https://github.com/CAOR-MINES-ParisTech/ukfm.


Getting Started
--------------------------------------------------------------------------------

1. Download the latest source code from
   [GitHub](https://github.com/CAOR-MINES-ParisTech/ukfm) (see Installation in
   the documentation).
2. Follow  the 2D robot localization for an introduction to the methodology. 


Examples
--------------------------------------------------------------------------------


Below is a list of examples from which the unscented Kalman filter on
parallelizable manifolds has been implemented.

- 2D robot localization (for introduction on simulated data and on real data).

- 3D Attitude estimation with an Inertial Measurement Unit (IMU) equipped with
  gyros, accelerometers and magnetometers.

-  3D inertial navigation on flat Earth with observations of known landmarks.

-  2D Simultaneous Localization And Mapping (SLAM).

-  IMU-GNSS sensor-fusion for a vehicle on the KITTI dataset.

-  Spherical pendulum example, where the state lives on the 2-sphere manifold.


Support
--------------------------------------------------------------------------------

Please, use the [GitHub issue
tracker](https://github.com/CAOR-MINES-ParisTech/ukfm) for questions, bug
reports, feature requests/additions, etc.


Acknowledgments
--------------------------------------------------------------------------------

The library was written by [Martin
Brossard](mailto:martin.brossard@mines-paristech.fr)^, [Axel
Barrau](mailto:axel.barrau@safrangroup.com)^ and [Silvère
Bonnabel](mailto:silvere.bonnabel@mines-paristech.fr)^.

^[MINES ParisTech](http://www.mines-paristech.eu/), PSL Research University,
Centre for Robotics, 60 Boulevard Saint-Michel, 75006 Paris, France.





