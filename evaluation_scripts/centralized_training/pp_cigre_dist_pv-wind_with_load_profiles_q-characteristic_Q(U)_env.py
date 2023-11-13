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



net = pn.create_cigre_network_mv(with_der="pv_wind")

res_multiplier = 1.5

gen_df = pd.read_csv("RESProfile.csv", sep=";")
    
res1_s_list = res_multiplier*gen_df['PV8']
res2_s_list = res_multiplier*gen_df['PV2']
res3_s_list = res_multiplier*gen_df['PV5']
res4_s_list = res_multiplier*gen_df['PV1']
res5_s_list = res_multiplier*gen_df['PV6']
res6_s_list = res_multiplier*gen_df['PV3']
res7_s_list = res_multiplier*gen_df['PV4']
res8_s_list = res_multiplier*gen_df['PV7']
res9_s_list = res_multiplier*gen_df['WP8']




def update_RES(step_counter):
    net.sgen['sn_mva'][0] = res1_s_list[step_counter]
    net.sgen['sn_mva'][1] = res2_s_list[step_counter]
    net.sgen['sn_mva'][2] = res3_s_list[step_counter]
    net.sgen['sn_mva'][3] = res4_s_list[step_counter]
    net.sgen['sn_mva'][4] = res5_s_list[step_counter]
    net.sgen['sn_mva'][5] = res6_s_list[step_counter]
    net.sgen['sn_mva'][6] = res7_s_list[step_counter]
    net.sgen['sn_mva'][7] = res8_s_list[step_counter]
    net.sgen['sn_mva'][8] = res9_s_list[step_counter]
    
    

bus_1_vm_pu_list = []
bus_2_vm_pu_list = []
bus_3_vm_pu_list = []
bus_4_vm_pu_list = []
bus_5_vm_pu_list = []
bus_6_vm_pu_list = []
bus_7_vm_pu_list = []
bus_8_vm_pu_list = []
bus_9_vm_pu_list = []
bus_10_vm_pu_list = []
bus_11_vm_pu_list = []
bus_12_vm_pu_list = []
bus_13_vm_pu_list = []


#episode_length = 2929
episode_average_error = 0

#last_vm_pu_list = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]




#https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial
#https://www.tensorflow.org/agents/api_docs/python/tf_agents/policies/PolicySaver

#comment normalization of vm_pu in env out to get exactly the same result as in training


eval_env = CigrePPEnv(True, "eval_env_log.db", [1,2,3,4,5,6,7,8,9,10,11,12], 0, "eval")
#tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)





def get_phase_shift_qu(vm_pu_list):
     
    phase_shift_list = []


    generator_counter = 0
    
    for i in range(len(net.sgen['name'])):
        #print(f"generator {generator_counter}")
        #print(f"generator bus {net.sgen['bus'][generator_counter]}")
        p_mw = 0
        q_mvar = 0
        
        current_generator_voltage = vm_pu_list[net.sgen['bus'][generator_counter]]
      
        if 0.96 <= current_generator_voltage <= 1.04:
            p_q_factor = (0.66/0.08) * abs(1.0 - current_generator_voltage)
            #print(p_q_factor)
            p_mw = math.sqrt(net.sgen['sn_mva'][generator_counter]**2/(1+(p_q_factor**2)))
            q_factor = 1
            if current_generator_voltage > 1.0 : q_factor = -1
            if p_q_factor != 0.0:
                q_mvar = q_factor * math.sqrt(net.sgen['sn_mva'][generator_counter]**2/(1+(1/(p_q_factor**2))))
            else:
                q_mvar = 0.0
                
        elif current_generator_voltage < 0.96:
            p_q_factor = 0.33
            p_mw = math.sqrt(net.sgen['sn_mva'][generator_counter]**2/(1+(p_q_factor**2)))
            q_mvar = math.sqrt(net.sgen['sn_mva'][generator_counter]**2/(1+(1/(p_q_factor**2))))
      
        elif 1.04 < current_generator_voltage:
            p_q_factor = 0.33
            p_mw = math.sqrt(net.sgen['sn_mva'][generator_counter]**2/(1+(p_q_factor**2)))
            q_mvar = -1*math.sqrt(net.sgen['sn_mva'][generator_counter]**2/(1+(1/(p_q_factor**2))))
        
        #print(f"u: {current_generator_voltage}")        
        #print(f"sn_mva: {net.sgen['sn_mva'][generator_counter]}")
        #print(f"p_mw: {p_mw}")
        #print(f"q_mvar: {q_mvar}")
            
        
        if(p_mw > 0):
            current_phase_shift = (math.degrees(math.atan(q_mvar/p_mw))+25.84)/51.68
            phase_shift_list.append(current_phase_shift)
            #print(f"phase_shift: {current_phase_shift}")
        else:
            phase_shift_list.append(0.0)
            #print(f"phase_shift: 0.0")
        
        generator_counter += 1
        
    
    return phase_shift_list





step_counter = 0
episode_average_vm_pu_deviation = 0


time_step = eval_env.reset()

#print(time_step.observation)

#scale observation after first reset because standard reset state is scaled to [0,1]
for i in range(15):
    time_step.observation[i] *= 2

while not time_step.is_last():


    update_RES(step_counter)
    #action_step = eager_py_policy.action(time_step)
    #print(action_step.action)
    #print(time_step.observation[0:15])
    action = get_phase_shift_qu(time_step.observation)
    #print(action)
    
    time_step = eval_env.step(action)
    #print(time_step.observation)
    
    

    vm_pu_list = time_step.observation[0:15]
    
    for i in range(len(vm_pu_list)):
        vm_pu_list[i] *= 2
    
    
    average_vm_pu_deviation = 0
    for current_bus_vm_pu in vm_pu_list[1:]:
        average_vm_pu_deviation += abs(1.0 - current_bus_vm_pu)
        
        
    average_vm_pu_deviation /= len(vm_pu_list[1:])
    episode_average_vm_pu_deviation += average_vm_pu_deviation
    step_counter += 1
    
    



print(f"step counter {step_counter}")

episode_average_vm_pu_deviation /= step_counter
print(f"average vm pu deviation in episode {episode_average_vm_pu_deviation}")




