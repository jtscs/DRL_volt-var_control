import sqlite3
from sqlite3 import Error

import matplotlib.pyplot as plt

#log_file_name = "..\\RL_Training\\CIGRE_MediumVoltage_benchmark\\SoftActorCritic\\Distributed_FullProfile\\results\\train7\\eval_env_log.db"
#log_file_name = "..\\RL_Training\\CIGRE_MediumVoltage_benchmark\\SoftActorCritic\\FullLoad_GenerationProfiles\\results\\Parametertests\\basic_lossLog\\eval_env_log.db"
log_file_name = "../RL_Training/CIGRE_MediumVoltage_benchmark/SoftActorCritic/FullLoad_GenerationProfiles/results/Parametertests/actorDistributionNet/eval_env_log.db"
log_file_name = "../RL_Training/Expanded_Training/Centralized_Regulation/ext_net_changes/eval_env_log.db"
#log_file_name = "../Trainings/SAC_cigre_pv-wind/results/train4/eval_env_log.db"
#log_file_name = "../Trainings/SAC_tappos-shiftangle_combined/results/train3/eval_env_log.db"

try:
    connection = sqlite3.connect(log_file_name)
    cursor = connection.cursor()
    # print(sqlite3.version)
except Error as e:
    print(e)
    quit()


step_counter = 0


def get_step(step_number):
    get_next_step_query = """SELECT * FROM steps WHERE step = ?"""
    try:
        cursor.execute(get_next_step_query, (step_number,))
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print(e)

    return []



step = get_step(step_counter)

episode_counter = 0
episode_step_counter = 0
episode_average_vm_pu_deviation = 0

average_vm_pu_deviation_per_episode_list = []
min_average_vm_pu_deviation = 1000
min_average_vm_pu_deviation_episode = 0

while step != []:
    #print(step[0][5])
    episode_average_vm_pu_deviation += step[0][11]
    #episode_average_vm_pu_deviation += abs(1.0 - step[0][5])
    
    #if(abs(step[0][12]) > 0.01):
    #    print(step[0])

    step_counter += 1
    episode_step_counter += 1

    if(episode_step_counter == 17476):
        #print("episode done")
        episode_counter += 1
        episode_average_vm_pu_deviation /= 17476
        if episode_average_vm_pu_deviation < min_average_vm_pu_deviation:
            min_average_vm_pu_deviation = episode_average_vm_pu_deviation
            min_average_vm_pu_deviation_episode = episode_counter
        average_vm_pu_deviation_per_episode_list.append(episode_average_vm_pu_deviation)
        episode_average_vm_pu_deviation = 0
        episode_step_counter = 0
        
        
    #if(step_counter % 2928 == 0):
    #    print("another episode done")

    step = get_step(step_counter)

    #if(step_counter > 999997):
    #    print(step)

print(f"best episode was {min_average_vm_pu_deviation_episode} with {min_average_vm_pu_deviation}")




fig, ax = plt.subplots()
    
fig.set_figwidth(9)

ax.plot(average_vm_pu_deviation_per_episode_list)

ax.set_xlabel(r'Episodenzahl', fontsize=10)
ax.set_ylabel(r'Durchschnittliche absolute Spannungsabweichung in pu', fontsize=10)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)

#plt.show()


plt.savefig('average_vm_pu_deviation_per_episode.png', dpi=350)
plt.savefig('average_vm_pu_deviation_per_episode.svg')

plt.clf()






plt.plot(average_vm_pu_deviation_per_episode_list)
plt.show()

connection.close()


"""if(self._plotting):
        plt.figure()
        plt.subplot(311)
        plt.plot(load_vm_pu_list)

        plt.subplot(312)
        plt.plot(load_p_list)

        plt.subplot(313)
        plt.plot(load_q_list)
        plt.show()

      load_vm_pu_list.clear()
      load_p_list.clear()
      load_q_list.clear()"""
