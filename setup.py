from setuptools import setup, find_packages
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))
subprocess.call('pandoc {0}/README.md -s -o {0}/README.rst'
                .format(here), shell=True)
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='causalmodels',
    version='0.3.3',
    description='Causal models in Python',
    long_description=long_description,
    url='http://github.com/roronya/causalmodels',
    author='roronya',
    author_email='roronya628@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords='causality machine	learning causalmodels',
    packages=find_packages(exclude=['tests*']),
    install_requires=['numpy', 'scipy', 'graphviz', 'tqdm', 'scikit-learn'],
)
