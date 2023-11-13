import sqlite3
from sqlite3 import Error

import matplotlib.pyplot as plt


#This script plots the average absolute voltage deviation over all evaluation episodes of a training


log_file_name = "../Centralized_Training/eval_env_log.db"


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

    episode_average_vm_pu_deviation += step[0][11]


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
        


    step = get_step(step_counter)


print(f"best episode was {min_average_vm_pu_deviation_episode} with {min_average_vm_pu_deviation}")




fig, ax = plt.subplots()
    
fig.set_figwidth(9)

ax.plot(average_vm_pu_deviation_per_episode_list)

ax.set_xlabel(r'Episode Count', fontsize=10)
ax.set_ylabel(r'Average absolute voltage deviation in pu', fontsize=10)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)

#plt.show()


plt.savefig('average_vm_pu_deviation_per_episode.png', dpi=350)
plt.savefig('average_vm_pu_deviation_per_episode.svg')

plt.clf()






plt.plot(average_vm_pu_deviation_per_episode_list)
plt.show()

connection.close()


