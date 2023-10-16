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
import os

from cigre_pv_wind_PPEnv import CigrePPEnv

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


#with tf.device("/cpu:0"):
eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy('actor_policy', eval_env.time_step_spec(), eval_env.action_spec())


time_step = eval_env.reset()

while not time_step.is_last():
    #print(time_step.observation)
    #with tf.device("/cpu:0"):
    action_step = eager_py_policy.action(time_step)
    #print(action_step.action)
    #for action in action_step.action: print(action)
    time_step = eval_env.step(action_step.action)
    #print(time_step.observation)


    vm_pu_list = time_step.observation[0:15]
    #print(vm_pu_list)
    
    
    average_vm_pu_deviation = 0
    for current_bus_vm_pu in vm_pu_list[1:]:
        average_vm_pu_deviation += abs(1.0 - current_bus_vm_pu*2)
        
        
    average_vm_pu_deviation /= len(vm_pu_list[1:])
    episode_average_vm_pu_deviation += average_vm_pu_deviation
    step_counter += 1

    #vm_pu_list refers to items in time_step, so by halving vm_pu list, voltages in time_step are also halved
    #for i in range(len(vm_pu_list)):
    #    vm_pu_list[i] /= 2


print(f"step counter {step_counter}")

episode_average_vm_pu_deviation /= step_counter
print(f"average vm pu deviation in episode {episode_average_vm_pu_deviation}")







