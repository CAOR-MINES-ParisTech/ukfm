UKF-M - Python Implementation
================================================================================

Download
--------------------------------------------------------------------------------
You can download the source code at
https://github.com/CAOR-MINES-ParisTech/ukfm.

Installation
--------------------------------------------------------------------------------
The Python package has been tested under Python 3.5 on a Ubuntu 16.04 machine.
To install:

1. Download the repo
```
    git clone https://github.com/CAOR-MINES-ParisTech/ukfm.git
```    
2. Install package requirement (numpy, matplotlib, etc)
```
    cd ukfm
    cd python
    pip install -r requirements.txt
```
3. Keep into the Python folder and run
```
    pip install .
```  
 or
```
    pip install -e .
```
The ``-e`` flag tells pip to install the package in-place, which lets you make
changes to the code without having to reinstall every time.

Get Started
--------------------------------------------------------------------------------
  
Follow the 2D robot localization tutorial in the documentation, which is
available at
[https://caor-mines-paristech.github.io/ukfm/]
(https://caor-mines-paristech.github.io/ukfm/).

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

