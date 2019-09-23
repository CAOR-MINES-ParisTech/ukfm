.. _installation:

Installation
================================================================================

You can download the source code at
https://github.com/CAOR-MINES-ParisTech/ukfm.

Python
--------------------------------------------------------------------------------
The Python package has been tested under Python 3.5 on a Ubuntu 16.04 machine.
To install:

1. Download the repo:

    .. highlight:: bash
    .. code-block:: bash

        git clone https://github.com/CAOR-MINES-ParisTech/ukfm.git
    
2. Install package requirement (numpy, matplotlib, etc):

    .. highlight:: bash
    .. code-block:: bash


        cd ukfm
        cd python
        pip install -r requirements.txt

3. Keep into the Python folder and run:

    .. highlight:: bash
    .. code-block:: bash

        pip install .

    or:

    .. highlight:: bash
    .. code-block:: bash

        pip install -e .

The ``-e`` flag tells pip to install the package in-place, which lets you make
changes to the code without having to reinstall every time.

Matlab
--------------------------------------------------------------------------------

The Matlab code has been tested with version R2019a without required any
particular toolbox. To install:

1. Download the repo:

    .. highlight:: bash
    .. code-block:: bash

        git clone https://github.com/CAOR-MINES-ParisTech/ukfm.git

2. Go to /my/directory/ukfm/matlab at the Matlab prompt and execute
   ``importukfm``.

3. You may save this path for your next Matlab sessions (via ``savepath``).



Documentation
--------------------------------------------------------------------------------

You need Sphinx to build the HTML documentation.

    .. highlight:: bash
    .. code-block:: bash

        cd docsource
        pip install sphinx
        make html
        open build/html/index.html

