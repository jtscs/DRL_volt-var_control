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


##########################################
#ATTENTION 
#
#This environment was slightly changed to work as evaluation env in the seperate eval process
# halving v_pu is commented because it is done in calling script if necessary
# a lot of evaluation is added
##########################################


def get_number_of_days_in_month(month):
  year = 2016
  return monthrange(year, month)[1]
  
  
#method to adjust profile length to different month length and comply with clock changes in march and october 
def get_month_profile_length(month):
  month_profile_length = get_number_of_days_in_month(month) * 24 * 4
  if month == 3 : month_profile_length -= 4
  if month == 10: month_profile_length += 4
  return month_profile_length
    
    
def get_month_profile_start_in_list(month):
  month_profile_start_counter = 0
  for i in range(1,month):
    month_profile_start_counter += get_month_profile_length(i)

  return month_profile_start_counter




#class to implement cigre medium voltage distribution net with pv and wind (https://pandapower.readthedocs.io/en/v2.3.0/networks/cigre.html) from pandas power as a learning environment for reinforcement learning
class CigrePPEnv(py_environment.PyEnvironment):

  #environment object constructor;
  #bool_log_env: true for logging, false to shut off logging
  #log_file_name: name of logging file, can be any string if bool_log_env if false
  #month_list: list for months that shall be used in training
  #train_mode: 0=use all months in month list for one episode; 1=use one random month in month list for one episode
  #name: string used for text output overview
  
  def __init__(self, bool_log_env, log_file_name, month_list, train_mode, name):
    #action are 9 shift angels of s_gen
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(9,), dtype=np.float32, minimum=0, maximum=1, name='action')
    #observation is all 15 bus voltages and all 9 previous shift angles
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(24,), dtype=np.float32, minimum=0, maximum=1, name='observation')
    #action and observation are normalized to values between 0 and 1. Usage of these values is adapted

    self._log_env = bool_log_env
    self._month_list = month_list
    self._train_mode = train_mode
    self._name = name
    
    if self._train_mode < 0 or self._train_mode > 1:
      print("train mode must be 0 or 1!")
      quit()


    self._state = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    self._step_counter = 0
    self._continuous_step_counter = 0
    self._episode_ended = False
    self._episode_average_error = 0
    
    if self._train_mode == 0:
     self._current_training_month = self._month_list[0]
    elif self._train_mode == 1:
      self._current_training_month = random.choice(self._month_list)
      
    self._month_profile_length = get_month_profile_length(self._current_training_month)
    self._current_profile_pointer = get_month_profile_start_in_list(self._current_training_month)
    self._month_iterator = 0
    self._mode0_month_step_counter = 0
    
    

    load_df = pd.read_csv("LoadProfile.csv", sep=";")

    self.load_time_list = load_df['time']
    self.load1_p_list = load_df['mv_semiurb_pload']
    self.load1_q_list = load_df['mv_semiurb_qload']
    self.load2_p_list = load_df['mv_urban_pload']
    self.load2_q_list = load_df['mv_urban_qload']
    self.load3_p_list = load_df['mv_comm_pload']
    self.load3_q_list = load_df['mv_comm_qload']
    self.load4_p_list = load_df['lv_rural3_pload']
    self.load4_q_list = load_df['lv_rural3_qload']
    self.load5_p_list = load_df['lv_urban6_pload']
    self.load5_q_list = load_df['lv_urban6_qload']
    self.load6_p_list = load_df['mv_semiurb_pload']
    self.load6_q_list = load_df['mv_semiurb_qload']
    self.load7_p_list = load_df['mv_urban_pload']
    self.load7_q_list = load_df['mv_urban_qload']
    self.load9_p_list = load_df['mv_comm_pload']
    self.load9_q_list = load_df['mv_comm_qload']
    
    self.load11_p_list = load_df['lv_rural1_pload']
    self.load11_q_list = load_df['lv_rural1_qload']
    self.load12_p_list = load_df['lv_rural2_pload']
    self.load12_q_list = load_df['lv_rural2_qload']
    self.load13_p_list = load_df['lv_semiurb4_pload']
    self.load13_q_list = load_df['lv_semiurb4_qload']
    self.load14_p_list = load_df['lv_semiurb5_pload']
    self.load14_q_list = load_df['lv_semiurb5_qload']

    self.load16_p_list = load_df['mv_rural_pload']
    self.load16_q_list = load_df['mv_rural_qload']
    self.load17_p_list = load_df['lv_semiurb4_pload']
    self.load17_q_list = load_df['lv_semiurb4_qload']
    
    
    gen_df = pd.read_csv("RESProfile.csv", sep=";")
    
    generation_multiplier = 1.5
    
    self.res_time_list = gen_df['time']
    self.res1_s_list = generation_multiplier*gen_df['PV8']
    self.res2_s_list = generation_multiplier*gen_df['PV2']
    self.res3_s_list = generation_multiplier*gen_df['PV5']
    self.res4_s_list = generation_multiplier*gen_df['PV1']
    self.res5_s_list = generation_multiplier*gen_df['PV6']
    self.res6_s_list = generation_multiplier*gen_df['PV3']
    self.res7_s_list = generation_multiplier*gen_df['PV4']
    self.res8_s_list = generation_multiplier*gen_df['PV7']
    self.res9_s_list = generation_multiplier*gen_df['WP8']
    

    self.net = pp.create_empty_network()
    self.init_net()

    if self._log_env:
      try:
        self._connection = sqlite3.connect(log_file_name)
        self._cursor = self._connection.cursor()
      except Error as e:
        print(e)

      create_steps_table_query = """ CREATE TABLE IF NOT EXISTS steps (
                                                  step integer PRIMARY KEY,
                                                  time string,
                                                  shift_angle_1 float,
                                                  shift_angle_2 float,
                                                  shift_angle_3 float,
                                                  shift_angle_4 float,
                                                  shift_angle_5 float,
                                                  shift_angle_6 float,
                                                  shift_angle_7 float,
                                                  shift_angle_8 float,
                                                  shift_angle_9 float,
                                                  bus_1_voltage float,
                                                  bus_2_voltage float,
                                                  bus_3_voltage float,
                                                  bus_4_voltage float,
                                                  bus_5_voltage float,
                                                  bus_6_voltage float,
                                                  bus_7_voltage float,
                                                  bus_8_voltage float,
                                                  bus_9_voltage float,
                                                  bus_10_voltage float,
                                                  bus_11_voltage float,
                                                  bus_12_voltage float,
                                                  bus_13_voltage float,
                                                  bus_14_voltage float,
                                                  trafo_1_load float,
                                                  trafo_2_load float,
                                                  line_0_load float,
                                                  line_1_load float,
                                                  line_2_load float,
                                                  line_3_load float,
                                                  line_4_load float,
                                                  line_5_load float,
                                                  line_6_load float,
                                                  line_7_load float,
                                                  line_8_load float,
                                                  line_9_load float,
                                                  line_10_load float,
                                                  line_11_load float,
                                                  line_12_load float,
                                                  line_13_load float,
                                                  line_14_load float,
                                                  generation_sum_s float,
                                                  average_vm_pu_deviation float
                                              ); """

      self.create_log_table(create_steps_table_query)
      
      self.episodeStartTime = time.perf_counter()



  def action_spec(self):
    return self._action_spec



  def observation_spec(self):
    return self._observation_spec
  
  
  
  #method to reset environment to default values
  def _reset(self):
    self._state = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    self._step_counter = 0
    self._episode_average_error = 0
    self._episode_ended = False
    self._mode0_month_step_counter = 0
    
    
    if self._train_mode == 0:
      self._current_training_month = self._month_list[0]
      print(f"current {self._name} month {self._current_training_month}")
    elif self._train_mode == 1:
      self._current_training_month = random.choice(self._month_list)
      print(f"current {self._name} month {self._current_training_month}")
      
    self._month_profile_length = get_month_profile_length(self._current_training_month)
    self._current_profile_pointer = get_month_profile_start_in_list(self._current_training_month)
    self._month_iterator = 0
    
    self.episodeStartTime = time.perf_counter()
    return ts.restart(np.array(self._state, dtype=np.float32))



  #method to do oe step in environment with one action and get one observation
  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start a new episode.
      return self.reset()

    #database logging made issues witout float conversion, converted floats might deliver slightly different results
    action_casted = [float(action[0]), float(action[1]), float(action[2]), float(action[3]), float(action[4]), float(action[5]), float(action[6]), float(action[7]), float(action[8])]
    
    
    # update state and vary load for current step
    self.update_loads(self._current_profile_pointer)
    self.update_RES(self._current_profile_pointer)
    
    #check if load_time_list and res_time_list are the same, only needed if profiles are changed to make sure they work together
    #if(self.load_time_list[self._current_profile_pointer] != self.res_time_list[self._current_profile_pointer]):
    #  print("load and res lists had a mismatch, quitting Training")
    #  quit()
    

    current_vm_pu_list = self.createAndRunNetSimulation(action_casted)
    

    #skip first element of vm_pu list since it is te bus connected to ext_grid whic has const voltage
    current_average_vm_pu_deviation = 0
    for vm_pu in current_vm_pu_list[1:]:
      current_average_vm_pu_deviation += abs(1.0-vm_pu)
    current_average_vm_pu_deviation /= len(current_vm_pu_list[1:])
    
    #normalize current_vm_pu_list to values between 0 and 1
    for i in range(len(current_vm_pu_list)):
        current_vm_pu_list[i] /= 2

    self._state = [*current_vm_pu_list , *action_casted]
    #print(f"state {self._state}")


    self._step_counter += 1
    self._continuous_step_counter += 1
    self._current_profile_pointer += 1
    self._mode0_month_step_counter += 1
    
    self._episode_average_error += current_average_vm_pu_deviation
    
    #multipy reward with 100 because vm_pu_deviation is very small, since vm_pu_deviation is to be minimized it must be converted tobe negative because the reward is maximized
    reward = -100 * current_average_vm_pu_deviation
    #print(f"reward {reward}")
    
    
    if(self._mode0_month_step_counter == (self._month_profile_length/2) or self._mode0_month_step_counter == self._month_profile_length):
      current_average_error_per_step = self._episode_average_error/self._step_counter
      print(f"Current average Error per step in episode {current_average_error_per_step}")
    
    
    
    if(self._train_mode == 0):
      if(self._mode0_month_step_counter >= self._month_profile_length):
        #if the iterator was at last month of month_list the iteration over all month is complete and the episode must end in mode 0
        if(self._month_iterator + 1 == len(self._month_list)):
           self._episode_ended = True
        else :
          self._month_iterator += 1
          self._current_training_month = self._month_list[self._month_iterator]
          print(f"current {self._name} month {self._current_training_month}")
          self._month_profile_length = get_month_profile_length(self._current_training_month)
          self._current_profile_pointer = get_month_profile_start_in_list(self._current_training_month)
          self._mode0_month_step_counter = 0
    
    elif(self._train_mode == 1):
      if(self._step_counter >= self._month_profile_length):
        self._episode_ended = True


    

    if self._episode_ended:
      if self._log_env:
        self.sql_commit()
        
      #print _episode_average_error to check how training is going
      self._episode_average_error /= self._step_counter
      print(f"Average Error per step in episode {self._episode_average_error}")
      episodeEndTime = time.perf_counter()
      print(f"Epsidoe Time: {episodeEndTime - self.episodeStartTime:0.4f} seconds")

      return ts.termination(np.array(self._state, dtype=np.float32), reward)
    else:
      return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=0.0)



  def init_net(self):
    self.net = pn.create_cigre_network_mv(with_der="pv_wind")


  def is_episode_finished(self):
    return self._episode_ended
  
  
  def update_loads(self, step_counter):
    self.net.load.p_mw[0] = 0
    self.net.load.q_mvar[0] = 0
    self.net.load.p_mw[1] = self.load1_p_list[step_counter]
    self.net.load.q_mvar[1] = self.load1_q_list[step_counter]
    self.net.load.p_mw[2] = self.load2_p_list[step_counter]
    self.net.load.q_mvar[2] = self.load2_q_list[step_counter]
    self.net.load.p_mw[3] = self.load3_p_list[step_counter]
    self.net.load.q_mvar[3] = self.load3_q_list[step_counter]
    self.net.load.p_mw[4] = self.load4_p_list[step_counter]
    self.net.load.q_mvar[4] = self.load4_q_list[step_counter]
    self.net.load.p_mw[5] = self.load5_p_list[step_counter]
    self.net.load.q_mvar[5] = self.load5_q_list[step_counter]
    self.net.load.p_mw[6] = self.load6_p_list[step_counter]
    self.net.load.q_mvar[6] = self.load6_q_list[step_counter]
    self.net.load.p_mw[7] = self.load7_p_list[step_counter]
    self.net.load.q_mvar[7] = self.load7_q_list[step_counter]
    self.net.load.p_mw[8] = 0
    self.net.load.q_mvar[8] = 0
    self.net.load.p_mw[9] = self.load9_p_list[step_counter]
    self.net.load.q_mvar[9] = self.load9_q_list[step_counter]
    self.net.load.p_mw[10] = 0
    self.net.load.q_mvar[10] = 0
    self.net.load.p_mw[11] = self.load11_p_list[step_counter]
    self.net.load.q_mvar[11] = self.load11_q_list[step_counter]
    self.net.load.p_mw[12] = self.load12_p_list[step_counter]
    self.net.load.q_mvar[12] = self.load12_q_list[step_counter]
    self.net.load.p_mw[13] = self.load13_p_list[step_counter]
    self.net.load.q_mvar[13] = self.load13_q_list[step_counter]
    self.net.load.p_mw[14] = self.load14_p_list[step_counter]
    self.net.load.q_mvar[14] = self.load14_q_list[step_counter]
    self.net.load.p_mw[15] = 0
    self.net.load.q_mvar[15] = 0
    self.net.load.p_mw[16] = self.load16_p_list[step_counter]
    self.net.load.q_mvar[16] = self.load16_q_list[step_counter]
    self.net.load.p_mw[17] = self.load17_p_list[step_counter]
    self.net.load.q_mvar[17] = self.load17_q_list[step_counter]
    
    self.current_load_s_sum = math.sqrt(self.net.load.p_mw[1]**2+self.net.load.q_mvar[1]**2)+math.sqrt(self.net.load.p_mw[2]**2+self.net.load.q_mvar[2]**2)+math.sqrt(self.net.load.p_mw[3]**2+self.net.load.q_mvar[3]**2)+math.sqrt(self.net.load.p_mw[4]**2+self.net.load.q_mvar[4]**2)+math.sqrt(self.net.load.p_mw[5]**2+self.net.load.q_mvar[5]**2)+math.sqrt(self.net.load.p_mw[6]**2+self.net.load.q_mvar[6]**2)+math.sqrt(self.net.load.p_mw[7]**2+self.net.load.q_mvar[7]**2)+math.sqrt(self.net.load.p_mw[9]**2+self.net.load.q_mvar[9]**2)+math.sqrt(self.net.load.p_mw[11]**2+self.net.load.q_mvar[11]**2)+math.sqrt(self.net.load.p_mw[12]**2+self.net.load.q_mvar[12]**2)+math.sqrt(self.net.load.p_mw[13]**2+self.net.load.q_mvar[13]**2)+math.sqrt(self.net.load.p_mw[14]**2+self.net.load.q_mvar[14]**2)+math.sqrt(self.net.load.p_mw[16]**2+self.net.load.q_mvar[16]**2)+math.sqrt(self.net.load.p_mw[17]**2+self.net.load.q_mvar[17]**2)

    #print(self.net.load)
    
    
    
  def update_RES(self, step_counter):
    self.net.sgen['sn_mva'][0] = self.res1_s_list[step_counter]
    self.net.sgen['sn_mva'][1] = self.res2_s_list[step_counter]
    self.net.sgen['sn_mva'][2] = self.res3_s_list[step_counter]
    self.net.sgen['sn_mva'][3] = self.res4_s_list[step_counter]
    self.net.sgen['sn_mva'][4] = self.res5_s_list[step_counter]
    self.net.sgen['sn_mva'][5] = self.res6_s_list[step_counter]
    self.net.sgen['sn_mva'][6] = self.res7_s_list[step_counter]
    self.net.sgen['sn_mva'][7] = self.res8_s_list[step_counter]
    self.net.sgen['sn_mva'][8] = self.res9_s_list[step_counter]
    
    #print(self.net.sgen)
  


  def createAndRunNetSimulation(self, action):

    #print(f"action {action}")
    #print(f"pre sgens {self.net.sgen}")

    generator_counter = 0
    for generator in self.net.sgen.name:
    
      current_phase_shift = action[generator_counter]*51.68-25.84
      #print(f"current_phase_shift {current_phase_shift}")

      self.net.sgen['p_mw'][generator_counter] = math.cos(math.radians(current_phase_shift)) * self.net.sgen['sn_mva'][generator_counter]
      self.net.sgen['q_mvar'][generator_counter] = math.sin(math.radians(current_phase_shift)) * self.net.sgen['sn_mva'][generator_counter]
      
      if(not math.isclose(self.net.sgen['sn_mva'][generator_counter], math.sqrt(self.net.sgen['p_mw'][generator_counter]**2+self.net.sgen['q_mvar'][generator_counter]**2), abs_tol=1e-10) and not (-0.0001 < self.net.sgen['sn_mva'][generator_counter] < 0)):
          print(f"self._continuous_step_counter: {self._continuous_step_counter}")
          print(f"load_time: {self.load_time_list[self._current_profile_pointer]}")
          print(f"p_mw: {self.net.sgen['p_mw'][generator_counter]}")
          print(f"q_mvar: {self.net.sgen['q_mvar'][generator_counter]}")
          print(f"p_mw: {self.net.sgen['p_mw'][generator_counter]}")
          print(f"sn_mva: {self.net.sgen['sn_mva'][generator_counter]}")
          print(f"calculated sn: {math.sqrt(self.net.sgen['p_mw'][generator_counter]**2+self.net.sgen['q_mvar'][generator_counter]**2)}")
          print(f"sn calculation was wrong")
          print("")
          
          
      generator_counter += 1

    #print(f"post sgens {self.net.sgen}")

    try:
      pp.runpp(self.net)
    except:
      print("could not calculate power flow")
      return 0

    #print(net.res_trafo)
    #print(net.res_line)
    #print(net.res_load)
    #print(net.res_ext_grid)
    #print(f"result {self.net.res_bus['vm_pu']}")
    
    
    generation_s_sum = self.net.sgen['sn_mva'][0]+self.net.sgen['sn_mva'][1]+self.net.sgen['sn_mva'][2]+self.net.sgen['sn_mva'][3]+self.net.sgen['sn_mva'][4]+self.net.sgen['sn_mva'][5]+self.net.sgen['sn_mva'][6]+self.net.sgen['sn_mva'][6]+self.net.sgen['sn_mva'][8]
    
    #if self.net.res_line['loading_percent'][0] > 100.0 and self.net.res_line['loading_percent'][1] > 100.0:
      #print(self.net.res_bus)
      #print(self.net.res_line)
      #print(self.net.sgen['p_mw'])
      #print(self.net.sgen['q_mvar'])
      #print(f"generation sum: {self.net.sgen['sn_mva'][0]+self.net.sgen['sn_mva'][1]+self.net.sgen['sn_mva'][2]+self.net.sgen['sn_mva'][3]+self.net.sgen['sn_mva'][4]+self.net.sgen['sn_mva'][5]+self.net.sgen['sn_mva'][6]+self.net.sgen['sn_mva'][6]+self.net.sgen['sn_mva'][8]}")
      #print(f"load sum: {self.current_load_s_sum}")
      
      
      #with open("overload_log.txt", "a") as logfile:
      #  logfile.write('step = {0}: {1}\n'.format(step, eval_results))
      
      
    current_vm_pu_list = self.net.res_bus['vm_pu']
    
    current_vm_pu_list = current_vm_pu_list.astype(np.float32)
    
    current_average_vm_pu_deviation = 0
    for vm_pu in current_vm_pu_list[1:]:
      current_average_vm_pu_deviation += abs(1.0-vm_pu)
    current_average_vm_pu_deviation /= len(current_vm_pu_list[1:])
    

    
    
    if self._log_env:
      log_query = ''' INSERT INTO steps(step,time, shift_angle_1, shift_angle_2, shift_angle_3, shift_angle_4, shift_angle_5, shift_angle_6, shift_angle_7, shift_angle_8, shift_angle_9, bus_1_voltage , bus_2_voltage , bus_3_voltage , bus_4_voltage, bus_5_voltage, bus_6_voltage, bus_7_voltage, bus_8_voltage, bus_9_voltage, bus_10_voltage, bus_11_voltage, bus_12_voltage, bus_13_voltage, bus_14_voltage, trafo_1_load, trafo_2_load, line_0_load, line_1_load, line_2_load, line_3_load, line_4_load, line_5_load, line_6_load, line_7_load, line_8_load, line_9_load, line_10_load, line_11_load, line_12_load, line_13_load, line_14_load, generation_sum_s, average_vm_pu_deviation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
      self.log_action(log_query, [self._continuous_step_counter, self.load_time_list[self._current_profile_pointer], action[0], action[1], action[2], action[3], action[4], action[5], action[6], action[7], action[8], self.net.res_bus['vm_pu'][1], self.net.res_bus['vm_pu'][2], self.net.res_bus['vm_pu'][3], self.net.res_bus['vm_pu'][4], self.net.res_bus['vm_pu'][5], self.net.res_bus['vm_pu'][6], self.net.res_bus['vm_pu'][7], self.net.res_bus['vm_pu'][8], self.net.res_bus['vm_pu'][9], self.net.res_bus['vm_pu'][10], self.net.res_bus['vm_pu'][11], self.net.res_bus['vm_pu'][12], self.net.res_bus['vm_pu'][13], self.net.res_bus['vm_pu'][14], self.net.res_trafo['loading_percent'][0], self.net.res_trafo['loading_percent'][1], self.net.res_line['loading_percent'][0], self.net.res_line['loading_percent'][1], self.net.res_line['loading_percent'][2], self.net.res_line['loading_percent'][3], self.net.res_line['loading_percent'][4], self.net.res_line['loading_percent'][5], self.net.res_line['loading_percent'][6], self.net.res_line['loading_percent'][7], self.net.res_line['loading_percent'][8], self.net.res_line['loading_percent'][9], self.net.res_line['loading_percent'][10], self.net.res_line['loading_percent'][11], self.net.res_line['loading_percent'][12], self.net.res_line['loading_percent'][13], self.net.res_line['loading_percent'][14], generation_s_sum, current_average_vm_pu_deviation])
    

    return (current_vm_pu_list)



  def create_log_table(self, create_table_sql):
    try:
        self._cursor.execute(create_table_sql)
    except Error as e:
        print(e)



  def log_action(self, log_query, log_data):
    try:
        self._cursor.execute(log_query, log_data)
        #self._connection.commit()
    except Error as e:
        print(e)
        
        
        
  def sql_commit(self):
    try:
        self._connection.commit()
    except Error as e:
        print(e)
