The Pendula
===========

This repository contains the codes used in the project

    The Pendula
    https://rcapobianco.github.io/notes/the_pendula/

The goal of this project is to present several classical pendulum systems,
ranging from integrable to chaotic, together with numerical simulations,
analytical solutions, and visualizations.

The codes are intended mainly for educational purposes, and are
written to accompany the detailed discussion in the notes.


Contents
--------

planar_pendulum/
    Nonlinear planar pendulum
    - numerical integration
    - small-angle approximation
    - exact solution using elliptic functions
    - phase space plots

double_pendulum/
    Double pendulum
    - numerical integration of the full nonlinear system
    - time series plots
    - energy conservation check

spherical_pendulum/
    Spherical pendulum
    - effective potential analysis
    - numerical solution of the equations of motion
    - analytical solution using Weierstrass elliptic functions
    - 3D orbit visualization
    - animation


Requirements
------------

Python codes require:

    numpy
    scipy
    matplotlib

The spherical pendulum code requires:

    Wolfram Mathematica / Wolfram Language


Usage
-----

Each folder contains an independent script that can be run directly.

Python scripts:

    python3 script_name.py

Mathematica script:

    Evaluate the notebook or run the .wl file inside Mathematica.


Notes
-----

These codes are not meant to be optimized libraries.
They are written to be clear and pedagogical, matching the formulas
presented in the notes.


Author
------

Rogerio Capobianco

Project page:
https://rcapobianco.github.io/notes/the_pendula/
