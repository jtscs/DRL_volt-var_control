# DRL_volt-var_control

This repo contains the training scripts to the volt-var control approaches presented in "Development approach of a volt-var control for inverter-coupled renewable energy plants using deep reinforcement learning" at the 22nd Wind & Solar Integration Workshop.

A few required python packages must be installed like tf-agents and pandapower.

The folder each contain the training scripts for one volt-var control approach. The training can be started by simply running the main.py file.
The training scripts produce a log file with some basic training data like the loss, a sql database with detailed data of all evaluation episodes and the best exported actor net.
