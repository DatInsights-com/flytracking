import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="flytracking",
    version="1.1.0",
    author="Yann S. Dufour",
    author_email="yann.dufour@datinsights.com",
    description=("Module to detect and track flies walking inside an arena."),
    license="MIT",
    keywords="detection tracking science",
    url="https://github.com/DatInsights-com/fly_tracking",
    packages=["fly_tracking"],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "opencv-python",
        "ultralytics",
        "plotnine",
        "lap",
    ],
    long_description=read("README"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
