# setup.py

from setuptools import setup, find_packages

setup(
    name="PoissonSolver",
    version="0.1",
    packages=find_packages(),
    author="Alberto Artoni",
    author_email="alberto.artoni1995@gmail.com",
    description="A package to solve the Poisson equation in 2D using the Finite Element Method",
    long_description=open("README.md").read(),
    url="https://github.com/yourusername/my_package",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)
