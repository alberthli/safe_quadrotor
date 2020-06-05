# AA203 MPC with CBFs Project

## Overview
Project: controlling a quadcopter with a multi-rate controller running a slow MPC linearization-based controller for high-level planning composed with a fast CBF nonlinear controller for low-level safe corrections.

## Environmental Management
Make sure you run the project in a conda virtual environment. To make an environment,
```
conda create --name NAMEOFYOURVENV python=3.8
```
Anytime after initializing the environment, activate before using project files:
```
conda activate NAMEOFYOURENV
```
To install the dependencies in the project, after activating the environment for the first time, run
```
pip install -r requirements.txt
```

## Organization
The project files are located in the proj directory.