# DRL_volt-var_control

This repo contains the training scripts to the volt-var control approaches presented in "Development approach of a volt-var control for inverter-coupled renewable energy plants using deep reinforcement learning" at the 22nd Wind & Solar Integration Workshop. The paper is the fiel 9B_2_WISO23_187_paper_Türk_Jurek.pdf

The folder each contain the training scripts for one volt-var control approach. The training can be started by simply running the main.py file.
The training scripts produce a log file with some basic training data like the loss, a sql database with detailed data of all evaluation episodes and the best exported actor net.

# Installation
Because the python package reverb is required, the software can currently be only installed on a linux operating system. It also requires python3.
To install the required python packages (tensorflow, tf_agents, numba, dm-reverb[tensorflow] and pandapower) you can use the install_commands.sh script. Simply use chmod u+x install_commands.sh to make it executable and the run ./install_commands.sh.

# Starting the training
To start the training, simply go into the training folder for the centralized or decentralized approach and run the main.py file with python3.
