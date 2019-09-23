
from setuptools import setup

setup(
    name='ukfm',
    version='1.0.0',
    description='Unscented Kalman Filtering on (Parallelizable) Manifolds in Python',
    author='Martin Brossard',
    author_email='martin.brossard@mines-paristech.fr',
    license='BSD-3',
    packages=['ukfm'],
    install_requires=['numpy', 'scipy', 'matplotlib']
)
