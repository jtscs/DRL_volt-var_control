import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting
import math
import simbench as sb
import pandas as pd
import copy
import matplotlib.pyplot as plt
import tf_agents.policies
import tensorflow as tf
from tf_agents.train.utils import spec_utils
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy

from cigre_pv_wind_PPEnv import CigrePPEnv
from simEnv_interface import simEnvInterface

episode_average_error = 0

#last_vm_pu_list = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]


#https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial
#https://www.tensorflow.org/agents/api_docs/python/tf_agents/policies/PolicySaver

#comment normalization of vm_pu in env out to get exactly the same result as in training


eval_env = CigrePPEnv(True, "eval_env_log.db", [1,2,3,4,5,6,7,8,9,10,11,12], 0, "eval")

#tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

#saved_policy = tf.compat.v2.saved_model.load('actor_policy')
#print(saved_policy)
#policy_state = saved_policy.get_initial_state(batch_size=3)
#print(policy_state)
#time_step = tf_eval_env.reset()
#print(time_step)

step_counter = 0
episode_average_vm_pu_deviation = 0

RES_count = 9


simEnv_interface_list = []
for i in range(1,RES_count+1):
    simEnv_interface_list.append(simEnvInterface('RES'+str(i), i-1, eval_env))
    
#print(simEnv_interface_list)

actor_policy_list = []

for i in range(1,RES_count+1):
    current_eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy('actor_policy_RES'+str(i), simEnv_interface_list[i-1].time_step_spec(), simEnv_interface_list[i-1].action_spec())
    actor_policy_list.append(current_eager_py_policy)
    
#print(actor_policy_list)


main_time_step = eval_env.reset()

time_step_list = []

for i in range(RES_count):
    time_step_list.append(simEnv_interface_list[i].reset())
    
#print(time_step_list)

while not main_time_step.is_last():

    action_list = [] 
    for index, actor_policy in enumerate(actor_policy_list):
        current_action_step = actor_policy.action(time_step_list[index])
        action_list.append(current_action_step.action[0])
        
    #print(action_list)
    #for action in action_list:
    #    print(action)
    
    main_time_step = eval_env.step(action_list)
    #print(time_step.observation)
    
    time_step_list.clear()
    
    for i in range(RES_count):
        time_step_list.append(simEnv_interface_list[i].step(action_list[i]))

    #print(time_step_list)

    vm_pu_list = main_time_step.observation[0:15]
    
    
    average_vm_pu_deviation = 0
    for current_bus_vm_pu in vm_pu_list[1:]:
        average_vm_pu_deviation += abs(1.0 - current_bus_vm_pu*2)
        
        
    average_vm_pu_deviation /= len(vm_pu_list[1:])
    episode_average_vm_pu_deviation += average_vm_pu_deviation
    step_counter += 1


print(f"step counter {step_counter}")

episode_average_vm_pu_deviation /= step_counter
print(f"average vm pu deviation in episode {episode_average_vm_pu_deviation}")







