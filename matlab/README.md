
UKF-M - Matlab Implementation
===============================================================================
We provide the equivalent Matlab code for designing UKF on (parallelizable)
manifolds, which is wholly independent from the python code.

Installation
--------------------------------------------------------------------------------

The Matlab code has been tested with version R2019a without requiring any 
particular toolbox. To install:

1.  Download the repo
```
git clone https://github.com/CAOR-MINES-ParisTech/ukfm.git
```

2. Go to /my/directory/ukfm/matlab at the Matlab prompt and execute
   ``importukfm``.

3. You may save this path for your next Matlab sessions (via ``savepath``).

Get Started
--------------------------------------------------------------------------------
Follow the 2D robot localization example (tutorial): in the Matlab prompt
execute
```
main_localization
```
  
Usage
--------------------------------------------------------------------------------
In contrast to Python, the code is implemented without class and has only
functions.

The file for designing an UKF are given in the ``ukfm`` folder and useful
geometry (Lie groups) functions are provided in the ``geometry`` folder.

We provide scripts for reproducing the examples and benchmarks respectively in
the ``examples`` and ``benchmarks`` folders. Models functions are organizedin
subfolder of the example folder: for e.g. the 2D robot localization model, see
in ``examples/localization``. You can use the Matlab publish tool for better
rendering and obtain the published scripts
[here](https://caor-mines-paristech.github.io/ukfm/matlab.html).

Function naming mimics the dot operator of class. To get the exponential of
$SE(3)` or the propagation function of the localization example, call
respectively ``se3_exp`` and ``localization_f``.

Citation
--------------------------------------------------------------------------------
If you use this project for your research, please please cite
```
@article{brossard2019Code,
  author={Martin Brossard and Axel Barrau and Silvère Bonnabe},
  title={{A Code for Unscented Kalman Filtering on Manifolds (UKF-M)}},
  year={2019},
}
```

