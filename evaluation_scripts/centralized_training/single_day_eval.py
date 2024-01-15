import pandas as pd
import math
import copy
from calendar import monthrange
import matplotlib.pyplot as plt
import sqlite3
from sqlite3 import Error


def get_number_of_days_in_month(month):
  year = 2016
  return monthrange(year, month)[1]
  
  
#method to adjust profile length to different month length and comply with clock changes in march and october 
def get_month_profile_length(month):
  month_profile_length = get_number_of_days_in_month(month) * 24 * 4
  if month == 3 : month_profile_length -= 4
  if month == 10: month_profile_length += 4
  return month_profile_length




generation_df = pd.read_csv("RESProfile.csv", sep=";")

gen_time = generation_df['time']
PV1_s_list = generation_df['PV1']
PV2_s_list = generation_df['PV2']
PV3_s_list = generation_df['PV3']
PV4_s_list = generation_df['PV4']
PV5_s_list = generation_df['PV5']
PV6_s_list = generation_df['PV6']
PV7_s_list = generation_df['PV7']
PV8_s_list = generation_df['PV8']

WP4_s_list = generation_df['WP4']
WP5_s_list = generation_df['WP5']
WP7_s_list = generation_df['WP7']
WP8_s_list = generation_df['WP8']
WP10_s_list = generation_df['WP10']




load_df = pd.read_csv("LoadProfile.csv", sep=";")

lv_rural1_p_list = load_df['lv_rural1_pload']
lv_rural1_q_list = load_df['lv_rural1_qload']
lv_rural2_p_list = load_df['lv_rural2_pload']
lv_rural2_q_list = load_df['lv_rural2_qload']
lv_rural3_p_list = load_df['lv_rural3_pload']
lv_rural3_q_list = load_df['lv_rural3_qload']
lv_semiurb4_p_list = load_df['lv_semiurb4_pload']
lv_semiurb4_q_list = load_df['lv_semiurb4_qload']
lv_semiurb5_p_list = load_df['lv_semiurb5_pload']
lv_semiurb5_q_list = load_df['lv_semiurb5_qload']
lv_urban6_p_list = load_df['lv_urban6_pload']
lv_urban6_q_list = load_df['lv_urban6_qload']

mv_rural_p_list = load_df['mv_rural_pload']
mv_rural_q_list = load_df['mv_rural_qload']
mv_semiurb_p_list = load_df['mv_semiurb_pload']
mv_semiurb_q_list = load_df['mv_semiurb_qload']
mv_urban_p_list = load_df['mv_urban_pload']
mv_urban_q_list = load_df['mv_urban_qload']
mv_comm_p_list = load_df['mv_comm_pload']
mv_comm_q_list = load_df['mv_comm_qload']

"""L0_A_p_list = load_df['L0-A_pload']
L0_A_q_list = load_df['L0-A_qload']
L2_M_p_list = load_df['L2-M_pload']
L2_M_q_list = load_df['L2-M_qload']

G0_A_p_list = load_df['G0-A_pload']
G0_A_q_list = load_df['G0-A_qload']
G0_M_p_list = load_df['G0-M_pload']
G0_M_q_list = load_df['G0-M_qload']

G3_M_p_list = load_df['G3-M_pload']
G3_M_q_list = load_df['G3-M_qload']
G3_H_p_list = load_df['G3-H_pload']
G3_H_q_list = load_df['G3-H_qload']

G4_A_p_list = load_df['G4-A_pload']
G4_A_q_list = load_df['G4-A_qload']
G4_M_p_list = load_df['G4-M_pload']
G4_M_q_list = load_df['G4-M_qload']"""




lv_rural1_s_list = []
lv_rural2_s_list = []
lv_rural3_s_list = []
lv_semiurb4_s_list = []
lv_semiurb5_s_list = []
lv_urban6_s_list = []

mv_rural_s_list = []
mv_semiurb_s_list = []
mv_urban_s_list = []
mv_comm_s_list = []

"""L0_A_s_list = []
L2_M_s_list = []

G0_A_s_list = []
G0_M_s_list = []

G3_M_s_list = []
G3_H_s_list = []

G4_A_s_list = []
G4_M_s_list = []"""


load_sum_list = []
generation_sum_list = []



sum_range = len(PV1_s_list)

generation_multiplier = 1.5

current_month = 1
next_month_end_index = get_month_profile_length(current_month)-1

day_counter = 1
step_per_day_counter = 0

current_day_s_difference = 0;
min_s_difference = 999999;
min_s_difference_day = "nope"
min_s_difference_iteration = 0
max_negative_s_differene = 0
max_negative_s_differene_day = "nope"
max_negative_s_differene_iteration = 0
max_s_differene = 0;
max_s_differene_day = "nope"
max_s_differene_iteration = 0

requiered_steps_per_day = 96


def get_required_steps_per_day(iteration):
    if iteration == 8255:
        #zeitvorstellung am 27.03.2016
        return 92
    if iteration == 29083:
        return 100
    return 96



for i in range(sum_range):

    lv_rural1_s_list.append(math.sqrt(lv_rural1_p_list[i]**2+lv_rural1_q_list[i]**2))
    lv_rural2_s_list.append(math.sqrt(lv_rural2_p_list[i]**2+lv_rural2_q_list[i]**2))
    lv_rural3_s_list.append(math.sqrt(lv_rural3_p_list[i]**2+lv_rural3_q_list[i]**2))
    lv_semiurb4_s_list.append(math.sqrt(lv_semiurb4_p_list[i]**2+lv_semiurb4_q_list[i]**2))
    lv_semiurb5_s_list.append(math.sqrt(lv_semiurb5_p_list[i]**2+lv_semiurb5_q_list[i]**2))
    lv_urban6_s_list.append(math.sqrt(lv_urban6_p_list[i]**2+lv_urban6_q_list[i]**2))

    mv_rural_s_list.append(math.sqrt(mv_rural_p_list[i]**2+mv_rural_q_list[i]**2))
    mv_semiurb_s_list.append(math.sqrt(mv_semiurb_p_list[i]**2+mv_semiurb_q_list[i]**2))
    mv_urban_s_list.append(math.sqrt(mv_urban_p_list[i]**2+mv_urban_q_list[i]**2))
    mv_comm_s_list.append(math.sqrt(mv_comm_p_list[i]**2+mv_comm_q_list[i]**2))

    
    
    
    
    load_sum_list.append(lv_rural1_s_list[i] + lv_rural2_s_list[i] + lv_rural3_s_list[i] + 2*lv_semiurb4_s_list[i] + lv_semiurb5_s_list[i] + lv_urban6_s_list[i] + mv_rural_s_list[i] + 2*mv_semiurb_s_list[i] + 2*mv_urban_s_list[i] + 2*mv_comm_s_list[i])
    generation_sum_list.append(generation_multiplier*(PV1_s_list[i] + PV2_s_list[i] + PV3_s_list[i] + PV4_s_list[i] + PV5_s_list[i] + PV6_s_list[i] + PV7_s_list[i] + PV8_s_list[i] + WP8_s_list[i]))
    
    current_day_s_difference += (load_sum_list[i] - generation_sum_list[i])
    
    step_per_day_counter += 1
        
    if step_per_day_counter == requiered_steps_per_day:
        #print(f"day end {gen_time[i]} reached at iteration {i} with {step_per_day_counter} steps")
        requiered_steps_per_day = get_required_steps_per_day(i)
        step_per_day_counter = 0
        
        if abs(current_day_s_difference) < min_s_difference:
            min_s_difference = abs(current_day_s_difference)
            min_s_difference_day = gen_time[i]
            min_s_difference_iteration = i
            #print(f"new min_s_difference day found {min_s_difference_day} with {min_s_difference}")
            
        if current_day_s_difference < max_negative_s_differene:
            max_negative_s_differene = current_day_s_difference
            max_negative_s_differene_day = gen_time[i]
            max_negative_s_differene_iteration = i
            #print(f"new max_negative_s_differene day found {max_negative_s_differene_day} with {max_negative_s_differene}")
        
        if current_day_s_difference > max_s_differene:
            max_s_differene = current_day_s_difference
            max_s_differene_day = gen_time[i]
            max_s_differene_iteration = i
            #print(f"new max_s_differene day found {max_s_differene_day} with {max_s_differene}")
       
        current_day_s_difference = 0
        
        
        
        
        
    if i >= next_month_end_index and current_month < 12:
        current_month += 1
        next_month_end_index += get_month_profile_length(current_month)
        print(f"end of month {gen_time[i]}")



print("")
print(f"generation/load ratio {sum(generation_sum_list)/sum(load_sum_list)}")
print("")
print(f"min_s_difference day {min_s_difference_day} with {min_s_difference}")
print(f"max_negative_s_differene day {max_negative_s_differene_day} with {max_negative_s_differene}")
print(f"max_s_differene day {max_s_differene_day} with {max_s_differene}")



def get_step(cursor, step_number):
    get_next_step_query = """SELECT * FROM steps WHERE step = ?"""
    try:
        cursor.execute(get_next_step_query, (step_number,))
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print(e)

    return []
    
def get_step_data(cursor, step_number, column_name):
    get_next_step_query = f"""SELECT {column_name} FROM steps WHERE step = ?"""
    try:
        cursor.execute(get_next_step_query, (step_number,))
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print(e)

    return []
    
    
    
    
def voltage_plotting(day_index, plot_name):

    actor_bus_vm_pu_list_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    ref_bus_vm_pu_list_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    ref_2_bus_vm_pu_list_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    
    actor_episode_average_vm_pu_deviation = 0
    ref_episode_average_vm_pu_deviation = 0
    ref_2_episode_average_vm_pu_deviation = 0
    
    
    bus_count = 14
    
    for i in range(day_index, day_index+96):
    
        ###FILL ACTOR DATA LISTS
        actor_rows = get_step(actor_cursor, i)
        
        actor_vm_pu_list = [actor_rows[0][11],actor_rows[0][12],actor_rows[0][13],actor_rows[0][14],actor_rows[0][15],actor_rows[0][16],actor_rows[0][17],actor_rows[0][18],actor_rows[0][19],actor_rows[0][20],actor_rows[0][21],actor_rows[0][22],actor_rows[0][23],actor_rows[0][24]]
        
        actor_episode_average_vm_pu_deviation += actor_rows[0][43]
        
        print(actor_rows[0][1])
        #print(actor_vm_pu_list)
        
        max_actor_vm_pu_difference = 0
        for j in range(2,10):
            if abs(actor_vm_pu_list[10] - actor_vm_pu_list[j]) > max_actor_vm_pu_difference:
                max_actor_vm_pu_difference = abs(actor_vm_pu_list[10] - actor_vm_pu_list[j])
        
        for bus_index in range(bus_count):
            actor_bus_vm_pu_list_list[bus_index].append(actor_vm_pu_list[bus_index])


        ###FILL ref DATA LISTS
        ref_rows = get_step(ref_cursor, i)
        
        ref_vm_pu_list = [ref_rows[0][11],ref_rows[0][12],ref_rows[0][13],ref_rows[0][14],ref_rows[0][15],ref_rows[0][16],ref_rows[0][17],ref_rows[0][18],ref_rows[0][19],ref_rows[0][20],ref_rows[0][21],ref_rows[0][22],ref_rows[0][23],ref_rows[0][24]]
        
        #print(ref_vm_pu_list)
        
        ref_episode_average_vm_pu_deviation += ref_rows[0][43]
        
        max_ref_vm_pu_difference = 0
        for k in range(2,10):
            if abs(ref_vm_pu_list[10] - ref_vm_pu_list[k]) > max_ref_vm_pu_difference:
                max_ref_vm_pu_difference = abs(ref_vm_pu_list[10] - ref_vm_pu_list[k])
        
        for bus_index in range(bus_count):
            ref_bus_vm_pu_list_list[bus_index].append(ref_vm_pu_list[bus_index])
            
            
        
        ###FILL ref 2 DATA LISTS
        ref_2_rows = get_step(ref_2_cursor, i)
        
        ref_2_vm_pu_list = [ref_2_rows[0][11],ref_2_rows[0][12],ref_2_rows[0][13],ref_2_rows[0][14],ref_2_rows[0][15],ref_2_rows[0][16],ref_2_rows[0][17],ref_2_rows[0][18],ref_2_rows[0][19],ref_2_rows[0][20],ref_2_rows[0][21],ref_2_rows[0][22],ref_2_rows[0][23],ref_2_rows[0][24]]
        
        
        #print(ref_2_vm_pu_list)
        
        ref_2_episode_average_vm_pu_deviation += ref_2_rows[0][43]
        
        
        max_ref_2_vm_pu_difference = 0
        for k in range(2,10):
            if abs(ref_2_vm_pu_list[10] - ref_2_vm_pu_list[k]) > max_ref_2_vm_pu_difference:
                max_ref_2_vm_pu_difference = abs(ref_2_vm_pu_list[10] - ref_2_vm_pu_list[k])
        
        
        for bus_index in range(bus_count):
            ref_2_bus_vm_pu_list_list[bus_index].append(ref_2_vm_pu_list[bus_index])
        
                
                
    actor_episode_average_vm_pu_deviation /= 96
    ref_episode_average_vm_pu_deviation /= 96
    ref_2_episode_average_vm_pu_deviation /= 96
    
    print("")
    
    print(f"actor_episode_average_vm_pu_deviation: {actor_episode_average_vm_pu_deviation}")
    print(f"ref_episode_average_vm_pu_deviation: {ref_episode_average_vm_pu_deviation}")
    print(f"ref_2_episode_average_vm_pu_deviation: {ref_2_episode_average_vm_pu_deviation}")
    
    print("")
    
    print(f"max_actor_vm_pu_difference {max_actor_vm_pu_difference}")
    print(f"max_ref_vm_pu_difference {max_ref_vm_pu_difference}")
    print(f"max_ref_2_vm_pu_difference {max_ref_2_vm_pu_difference}")
    
    print("\n\n")
    
    fig, ax = plt.subplots()
    
    fig.set_figwidth(9)
    
    ax.plot(actor_bus_vm_pu_list_list[0], linestyle='solid', color='blue', label='Centralized ANN bus 1')
    ax.plot(ref_bus_vm_pu_list_list[0], linestyle='solid', color='orange', label='Decentralized ANN bus 1')
    ax.plot(ref_2_bus_vm_pu_list_list[0], linestyle='solid', color='green', label='Q(U) with limited dynamic bus 1')
    #ax.plot(actor_bus_vm_pu_list_list[2], label='Bus 2')
    #ax.plot(actor_bus_vm_pu_list_list[3], label='Bus 3')
    #ax.plot(actor_bus_vm_pu_list_list[4], label='Bus 4')
    #ax.plot(actor_bus_vm_pu_list_list[5], label='Bus 5')
    #ax.plot(actor_bus_vm_pu_list_list[6], label='Bus 6')
    #ax.plot(actor_bus_vm_pu_list_list[7], label='Bus 7')
    #ax.plot(actor_bus_vm_pu_list_list[8], label='Bus 8')
    #ax.plot(actor_bus_vm_pu_list_list[9], label='Bus 9')
    ax.plot(actor_bus_vm_pu_list_list[10], linestyle='dotted', color='blue', label='Centralized ANN bus 11')
    ax.plot(ref_bus_vm_pu_list_list[10], linestyle='dotted', color='orange', label='Decentralized ANN bus 11')
    ax.plot(ref_2_bus_vm_pu_list_list[10], linestyle='dotted', color='green', label='Q(U) with limited dynamic bus 11')
    #ax.plot(actor_bus_vm_pu_list_list[11], label='Bus 11')
    #ax.plot(actor_bus_vm_pu_list_list[12], label='Bus 12')
    #ax.plot(actor_bus_vm_pu_list_list[13], label='Bus 13')
    legend = ax.legend(loc='best', fontsize=14)
    plt.xlabel('steps', fontsize=14)
    plt.ylabel('Bus voltage in pu', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_ylim([0.93, 1.05])
    #plt.title('Bus voltages in in day')
    
    
    plt.savefig(plot_name+'.png', dpi=600)
    plt.savefig(plot_name+'.svg')
    
    plt.show()

    
    

#log_file_name_actor = "newpc_lossLog2_eval_env_log.db"
#log_file_name_ref = "fixed_phi_-18_eval_env_log.db"

log_file_name_actor = "centralized_eval_env_log.db"
log_file_name_ref = "qu_eval_env_log.db"
log_file_name_ref_2 = "centralized_eval_env_log.db"


try:
    actor_connection = sqlite3.connect(log_file_name_actor)
    actor_cursor = actor_connection.cursor()
    # print(sqlite3.version)
except Error as e:
    print(e)
    quit()
    
actor_cursor.execute("""SELECT COUNT(*) FROM steps""", )
rows = actor_cursor.fetchall()
actor_db_length = rows[0][0]

print(f"Actor Database Length: {actor_db_length}")



try:
    ref_connection = sqlite3.connect(log_file_name_ref)
    ref_cursor = ref_connection.cursor()
    # print(sqlite3.version)
except Error as e:
    print(e)
    quit()
    
ref_cursor.execute("""SELECT COUNT(*) FROM steps""", )
rows = ref_cursor.fetchall()
ref_db_length = rows[0][0]

print(f"Reference Database Length: {ref_db_length}")


try:
    ref_2_connection = sqlite3.connect(log_file_name_ref_2)
    ref_2_cursor = ref_2_connection.cursor()
    # print(sqlite3.version)
except Error as e:
    print(e)
    quit()
    
ref_2_cursor.execute("""SELECT COUNT(*) FROM steps""", )
rows = ref_2_cursor.fetchall()
ref_2_db_length = rows[0][0]

print(f"Reference 2 Database Length: {ref_2_db_length}")


voltage_plotting(min_s_difference_iteration-95, "actor_phi-18_ausgeglichenen_Last_Spannungsplots")
voltage_plotting(max_negative_s_differene_iteration-95, "actor_phi-18_maximaler_Erzeugungsleistung_Spannungsplots")
voltage_plotting(max_s_differene_iteration-95, "actor_phi-18_maximale_Lastleistung_Spannungsplots")
