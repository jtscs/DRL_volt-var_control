import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import math
import random

import sqlite3
from sqlite3 import Error

import time

from calendar import monthrange

from cigre_pv_wind_PPEnv import CigrePPEnv



#class to implement cigre medium voltage distribution net with pv and wind (https://pandapower.readthedocs.io/en/v2.3.0/networks/cigre.html) from pandas power as a learning environment for reinforcement learning
class simEnvInterface(py_environment.PyEnvironment):

  #environment object constructor;
  #bool_log_env: true for logging, false to shut off logging
  #log_file_name: name of logging file, can be any string if bool_log_env if false
  #month_list: list for months that shall be used in training
  #train_mode: 0=use all months in month list for one episode; 1=use one random month in month list for one episode
  #name: string used for text output overview
  
  def __init__(self, name, id , mainSimEnv):
    #action are 9 shift angels of s_gen
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1, name='action')
    #observation is all 15 bus voltages and all 9 previous shift angles
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=0, maximum=1, name='observation')
    #action and observation are normalized to values between 0 and 1. Usage of these values is adapted

    self.name = name
    self.id = id

    self.net = pn.create_cigre_network_mv(with_der="pv_wind")
    self.mainSimEnvHandle = mainSimEnv

    self._state = [0.5,0.5]
    self._step_counter = 0

    self._episode_ended = False
    self._episode_average_error = 0




  def action_spec(self):
    return self._action_spec


  def observation_spec(self):
    return self._observation_spec


  
  #method to reset environment to default values
  def _reset(self):
    self._state = [0.5,0.5]
    self._step_counter = 0
    self._episode_average_error = 0
    self._episode_ended = False
    
    

    return ts.restart(np.array(self._state, dtype=np.float32))



  #method to do oe step in environment with one action and get one observation
  def _step(self, action):
    
    #print(f"{self.name} current action {action}")

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start a new episode.
      return self.reset()


    fullstate = self.mainSimEnvHandle.get_state()

    currentGeneratorVoltage = fullstate[self.net.sgen['bus'][self.id]]
    currentPhaseShift = fullstate[15 + self.id]
    
    self._state = [currentGeneratorVoltage, currentPhaseShift]
    #print(f"{self.name} state: {self._state}")
    
    #unnormalization of voltage to calculate voltage deviation and reward
    currentGeneratorVoltage *= 2
    
    
    current_average_vm_pu_deviation = abs(1.0-currentGeneratorVoltage)
    
    #print(f"{self.name}:  voltage={currentGeneratorVoltage} ; phase_shift={currentPhaseShift}")


    self._episode_average_error += current_average_vm_pu_deviation

    reward = -100 * current_average_vm_pu_deviation

    self._step_counter += 1
    
    self._episode_ended = self.mainSimEnvHandle.is_episode_finished()

    if self._episode_ended:
      self._episode_average_error /= self._step_counter
      print(f"{self.name} episode ended with {self._step_counter} steps")
      print(f"{self.name} average episode error {self._episode_average_error}")
      return ts.termination(np.array(self._state, dtype=np.float32), reward)
    else:
      return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=0.0)

