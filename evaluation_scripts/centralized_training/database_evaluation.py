import sqlite3
from sqlite3 import Error
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import PercentFormatter
import pandas as pd

log_file_name_actor = "centralized_eval_env_log.db"
log_file_name_ref = "qu_eval_env_log.db"


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
WP8_s_list = generation_df['WP8']





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
    
    
    
    
def check_db_length_compatibility(db_1_length, db_2_length):
    if(db_1_length != db_2_length):
        print("Databases have different length!\n")
        quit()
    else:
        print("Databases have the same length and are compatible\n")

def check_gen_s_sum():
    s_sum_check_valid = True
    for i in range(actor_db_length):
        actor_gen_s_sum = get_step_data(actor_cursor, i, "generation_sum_s")
        ref_gen_s_sum = get_step_data(ref_cursor, i, "generation_sum_s")
        
        if(actor_gen_s_sum != ref_gen_s_sum):
            print(f"database have different s_sum in step {i}")
            print(f"actor_gen_s_sum: {actor_gen_s_sum}")
            print(f"ref_gen_s_sum: {ref_gen_s_sum}")
            
            s_sum_check_valid = False
            
    if (s_sum_check_valid):
        print("Both databases have the same generation s_sum in all entries, check valid\n")
    else:
        print("s sum check failed, see entries above\n")
        quit()
        
        
def voltage_comparison():

    actor_bus_vm_pu_list_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    #overvoltage meaning voltage above 1.0 pu
    actor_bus_overvolt_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    #overvoltage violation meaning voltage above 1.05 pu
    actor_bus_overvolt_violation_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    #undervoltage violation meaning voltage lower than .95 pu
    actor_bus_undervolt_violation_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    
    ref_bus_vm_pu_list_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    #overvoltage meaning voltage above 1.0 pu
    ref_bus_overvolt_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    #overvoltage violation meaning voltage above 1.05 pu
    ref_bus_overvolt_violation_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    #undervoltage violation meaning voltage lower than .95 pu
    ref_bus_undervolt_violation_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    
    actor_episode_average_vm_pu_deviation = 0
    ref_episode_average_vm_pu_deviation = 0
    
    actor_with_worse_voltage_count = 0
    actor_created_overvolt_violation_count = 0
    actor_created_undervolt_violation_count = 0
    
    bus_count = 14
    
    for i in range(actor_db_length):
    
        ###FILL ACTOR DATA LISTS
        actor_rows = get_step(actor_cursor, i)
        
        actor_vm_pu_list = [actor_rows[0][11],actor_rows[0][12],actor_rows[0][13],actor_rows[0][14],actor_rows[0][15],actor_rows[0][16],actor_rows[0][17],actor_rows[0][18],actor_rows[0][19],actor_rows[0][20],actor_rows[0][21],actor_rows[0][22],actor_rows[0][23],actor_rows[0][24]]
        
        actor_episode_average_vm_pu_deviation += actor_rows[0][43]
        
        for bus_index in range(bus_count):
        
            actor_bus_vm_pu_list_list[bus_index].append(actor_vm_pu_list[bus_index])
            if actor_vm_pu_list[bus_index] > 1.0 : actor_bus_overvolt_count_list[bus_index] += 1    
            if actor_vm_pu_list[bus_index] > 1.05 : actor_bus_overvolt_violation_count_list[bus_index] += 1
            if actor_vm_pu_list[bus_index] < 0.95 : actor_bus_undervolt_violation_count_list[bus_index] += 1

        
        ###FILL ref DATA LISTS
        ref_rows = get_step(ref_cursor, i)
        
        ref_vm_pu_list = [ref_rows[0][11],ref_rows[0][12],ref_rows[0][13],ref_rows[0][14],ref_rows[0][15],ref_rows[0][16],ref_rows[0][17],ref_rows[0][18],ref_rows[0][19],ref_rows[0][20],ref_rows[0][21],ref_rows[0][22],ref_rows[0][23],ref_rows[0][24]]
        
        ref_episode_average_vm_pu_deviation += ref_rows[0][43]
        
        
        for bus_index in range(bus_count):
        
            ref_bus_vm_pu_list_list[bus_index].append(ref_vm_pu_list[bus_index])
            if ref_vm_pu_list[bus_index] > 1.0 : ref_bus_overvolt_count_list[bus_index] += 1    
            if ref_vm_pu_list[bus_index] > 1.05 : ref_bus_overvolt_violation_count_list[bus_index] += 1
            if ref_vm_pu_list[bus_index] < 0.95 : ref_bus_undervolt_violation_count_list[bus_index] += 1
        
        
        
        ###COMPARE ACTOR AND ref
        for bus_index in range(bus_count):
            if(abs(actor_vm_pu_list[bus_index] - 1.0) > abs(ref_vm_pu_list[bus_index] - 1.0)):
                #print(f"actor has worse voltage on line {bus_index} at time step {i}")
                #print(f"actor bus voltage: {actor_vm_pu_list[bus_index]} ; ref bus voltage: {ref_vm_pu_list[bus_index]}")
                actor_with_worse_voltage_count += 1
                
                
            if(actor_vm_pu_list[bus_index] > 1.05 and ref_vm_pu_list[bus_index] <= 1.05):
                actor_created_overvolt_violation_count += 1
                #print("actor created an overvolt situation where the static curve did not")
                
            if(actor_vm_pu_list[bus_index] < 0.95 and ref_vm_pu_list[bus_index] >= 0.95):
                actor_created_undervolt_violation_count += 1
                #print("actor created an undervolt violation where the static curve did not")
                
                
    actor_episode_average_vm_pu_deviation /= actor_db_length
    ref_episode_average_vm_pu_deviation /= ref_db_length
    
    print("")
    
    print(f"actor_episode_average_vm_pu_deviation: {actor_episode_average_vm_pu_deviation}")
    print(f"ref_episode_average_vm_pu_deviation: {ref_episode_average_vm_pu_deviation}")
    
    print("\n\n")
    
    print("Actor Highvolt Percentage")
    for index, overvolt_count in enumerate(actor_bus_overvolt_count_list): actor_bus_overvolt_count_list[index]/= actor_db_length
    print(actor_bus_overvolt_count_list)
    print("")
    
    print("Actor Overvolt Violations")
    actor_sum_overvolt_violations = sum(actor_bus_overvolt_violation_count_list)
    print(actor_sum_overvolt_violations)
    for index, actor_overvolt_violation_count in enumerate(actor_bus_overvolt_violation_count_list):
        if(actor_overvolt_violation_count != 0):
            print(f"{actor_overvolt_violation_count} overvoltage violations on bus {index}")
    
    print("Actor Undervolt Violations")
    actor_sum_undervolt_violations = sum(actor_bus_undervolt_violation_count_list)
    print(actor_sum_undervolt_violations)
    for index, actor_undervolt_violation_count in enumerate(actor_bus_undervolt_violation_count_list):
        if(actor_undervolt_violation_count != 0):
            print(f"{actor_undervolt_violation_count} undervoltage violations on bus {index}")
    
    print("\n\n")
    
    print("Reference Highvolt Percentage")
    for index, overvolt_count in enumerate(ref_bus_overvolt_count_list): ref_bus_overvolt_count_list[index]/= ref_db_length
    print(ref_bus_overvolt_count_list)
    print("")
    
    print("Reference Overvolt Violations")
    ref_sum_overvolt_violations = sum(ref_bus_overvolt_violation_count_list)
    print(ref_sum_overvolt_violations)
    for index, ref_overvolt_violation_count in enumerate(ref_bus_overvolt_violation_count_list):
        if(ref_overvolt_violation_count != 0):
            print(f"{ref_overvolt_violation_count} overvoltage violations on bus {index}")
    
    print("Reference Undervolt Violations")
    ref_sum_undervolt_violations = sum(ref_bus_undervolt_violation_count_list)
    print(ref_sum_undervolt_violations)
    for index, ref_undervolt_violation_count in enumerate(ref_bus_undervolt_violation_count_list):
        if(ref_undervolt_violation_count != 0):
            print(f"{ref_undervolt_violation_count} undervoltage violations on bus {index}")
    
    
    print("")
    if(ref_sum_overvolt_violations > 0):
        print("Overvolt Violation Comparison")
        overvolt_violation_reduction_percentage = (ref_sum_overvolt_violations-actor_sum_overvolt_violations)/ref_sum_overvolt_violations
        print(f"actor reduced overvolt violations by {overvolt_violation_reduction_percentage*100} percent\n")
    
    if(ref_sum_undervolt_violations > 0):
        print("Undervolt Violation Comparison")
        undervolt_violation_reduction_percentage = (ref_sum_undervolt_violations-actor_sum_undervolt_violations)/ref_sum_undervolt_violations
        print(f"actor reduced undervolt violations by {undervolt_violation_reduction_percentage*100} percent\n")
    
    print("")
    print("Actor with worse Voltage Percentage")
    actor_with_worse_voltage_count /= (bus_count * actor_db_length)
    print(actor_with_worse_voltage_count)
    
    print("")
    print("Actor Overvolt violations where static curve was in limits")
    print(actor_created_overvolt_violation_count)
    
    print("")
    print("Actor Undervolt violations where static curve was in limits")
    print(actor_created_undervolt_violation_count)
    
    
    print("\n\n\n")

    



def load_comparison():

    actor_trafo_load_sum_list = [0,0]
    actor_line_load_sum_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    #overload -> load above 100%
    actor_trafo_overload_count_list = [0,0]
    actor_line_overload_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    
    ref_trafo_load_sum_list = [0,0]
    ref_line_load_sum_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    #overload -> load above 100%
    ref_trafo_overload_count_list = [0,0]
    ref_line_overload_count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    actor_shift_angles_all = []
    ref_shift_angles_all = []
    

    trafo_count = 2
    line_count = 15
    
    for i in range(actor_db_length):
    
        ###FILL ACTOR DATA LISTS
        actor_rows = get_step(actor_cursor, i)
        
        actor_trafo_loads = [actor_rows[0][25],actor_rows[0][26]]
        actor_line_loads = [actor_rows[0][27],actor_rows[0][28],actor_rows[0][29],actor_rows[0][30],actor_rows[0][31],actor_rows[0][32],actor_rows[0][33],actor_rows[0][34],actor_rows[0][35],actor_rows[0][36],actor_rows[0][37],actor_rows[0][38],actor_rows[0][39],actor_rows[0][40],actor_rows[0][41]]
        
        
        for trafo_index in range(trafo_count):
            actor_trafo_load_sum_list[trafo_index]+=(actor_trafo_loads[trafo_index])
            if actor_trafo_loads[trafo_index] > 100.0 : actor_trafo_overload_count_list[trafo_index] += 1
            
        for line_index in range(line_count):
            actor_line_load_sum_list[line_index]+=(actor_line_loads[line_index])
            if actor_line_loads[line_index] > 100.0 : actor_line_overload_count_list[line_index] += 1  

        
        ###FILL ref DATA LISTS
        ref_rows = get_step(ref_cursor, i)
        
        ref_trafo_loads = [ref_rows[0][25],ref_rows[0][26]]
        ref_line_loads = [ref_rows[0][27],ref_rows[0][28],ref_rows[0][29],ref_rows[0][30],ref_rows[0][31],ref_rows[0][32],ref_rows[0][33],ref_rows[0][34],ref_rows[0][35],ref_rows[0][36],ref_rows[0][37],ref_rows[0][38],ref_rows[0][39],ref_rows[0][40],ref_rows[0][41]]
        
        
        for trafo_index in range(trafo_count):
            ref_trafo_load_sum_list[trafo_index]+=(ref_trafo_loads[trafo_index])
            if ref_trafo_loads[trafo_index] > 100.0 : ref_trafo_overload_count_list[trafo_index] += 1
            
        for line_index in range(line_count):
            ref_line_load_sum_list[line_index]+=(ref_line_loads[line_index])
            if ref_line_loads[line_index] > 100.0 : ref_line_overload_count_list[line_index] += 1
            
            
        actor_vm_pu_list = [actor_rows[0][11],actor_rows[0][12],actor_rows[0][13],actor_rows[0][14],actor_rows[0][15],actor_rows[0][16],actor_rows[0][17],actor_rows[0][18],actor_rows[0][19],actor_rows[0][20],actor_rows[0][21],actor_rows[0][22],actor_rows[0][23],actor_rows[0][24]]
        
        ref_vm_pu_list = [ref_rows[0][11],ref_rows[0][12],ref_rows[0][13],ref_rows[0][14],ref_rows[0][15],ref_rows[0][16],ref_rows[0][17],ref_rows[0][18],ref_rows[0][19],ref_rows[0][20],ref_rows[0][21],ref_rows[0][22],ref_rows[0][23],ref_rows[0][24]]
            
        actor_shift_angles = [actor_rows[0][2],actor_rows[0][3],actor_rows[0][4],actor_rows[0][5],actor_rows[0][6],actor_rows[0][7],actor_rows[0][8],actor_rows[0][9],actor_rows[0][10]]
        for index, shift_angle in enumerate(actor_shift_angles) : actor_shift_angles[index] = actor_shift_angles[index]*51.68-25.84

        
        ref_shift_angles = [ref_rows[0][2],ref_rows[0][3],ref_rows[0][4],ref_rows[0][5],ref_rows[0][6],ref_rows[0][7],ref_rows[0][8],ref_rows[0][9],ref_rows[0][10]]
        for index, shift_angle in enumerate(ref_shift_angles) : ref_shift_angles[index] = ref_shift_angles[index]*51.68-25.84
        
        if PV8_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[0])
            ref_shift_angles_all.append(ref_shift_angles[0])
        if PV2_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[1])
            ref_shift_angles_all.append(ref_shift_angles[1])
        if PV5_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[2])
            ref_shift_angles_all.append(ref_shift_angles[2])
        if PV1_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[3])
            ref_shift_angles_all.append(ref_shift_angles[3])
        if PV6_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[4])
            ref_shift_angles_all.append(ref_shift_angles[4])
        if PV3_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[5])
            ref_shift_angles_all.append(ref_shift_angles[5])            
        if PV4_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[6])
            ref_shift_angles_all.append(ref_shift_angles[6])
        if PV7_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[7])
            ref_shift_angles_all.append(ref_shift_angles[7])            
        if WP8_s_list[i] != 0.0:
            actor_shift_angles_all.append(actor_shift_angles[8])
            ref_shift_angles_all.append(ref_shift_angles[8])
            
        """for line_index in range(line_count):
            if(actor_line_loads[line_index] > 100.0 and ref_line_loads[line_index] <= 100.0):
                print("#########################################################")
                print(f"actor created overload on line {line_index}")
                print("#########################################################")
                print(f"actor load {actor_line_loads[line_index]}")
                print(f"ref load {ref_line_loads[line_index]}")
                print("")
                print("actor vm pu list")
                print(actor_vm_pu_list)
                print("")
                print("ref vm pu list")
                print(ref_vm_pu_list)
                print("")
                print("actor shift angles")
                print(actor_shift_angles)
                print("")
                print("ref shift angles")
                print(ref_shift_angles)
                print("\n")"""
                
            
    
    
    
    
    print("Actor Trafo Average Load")
    for index, overload_sum in enumerate(actor_trafo_load_sum_list): actor_trafo_load_sum_list[index]/= actor_db_length
    print(actor_trafo_load_sum_list)
    print("")
            

    print("Actor Line Average Load")
    for index, overload_sum in enumerate(actor_line_load_sum_list): actor_line_load_sum_list[index]/= actor_db_length
    print(actor_line_load_sum_list)
    print("")
    
    print("Actor Trafo Overload Count")
    print(actor_trafo_overload_count_list)
    print("")
           
    print("Actor Line Overload Count")
    print(actor_line_overload_count_list)
    print("")
    
    print("Actor Line Overload Sum")
    print(sum(actor_line_overload_count_list))
    print("")
    
    
    
    print("Reference Trafo Average Load")
    for index, overload_sum in enumerate(ref_trafo_load_sum_list): ref_trafo_load_sum_list[index]/= actor_db_length
    print(ref_trafo_load_sum_list) 
    print("")

    print("Reference Line Average Load")
    for index, overload_sum in enumerate(ref_line_load_sum_list): ref_line_load_sum_list[index]/= actor_db_length
    print(ref_line_load_sum_list)
    print("")

    print("Reference Trafo Overload Count")
    print(ref_trafo_overload_count_list)
    print("")
           
    print("Reference Line Overload Count")
    print(ref_line_overload_count_list)    
    print("")
    
    print("Reference Line Overload Sum")
    print(sum(ref_line_overload_count_list))
    print("")
    
    
        
    
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    
    fig.set_figwidth(9)
    
    n_bins = 40

    
    # We can also normalize our inputs by the total number of counts
    axs[0].hist(actor_shift_angles_all, bins=n_bins, density=True)
    axs[1].hist(ref_shift_angles_all, bins=n_bins, density=True)

    # Now we format the y-axis to display percentage
    axs[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    
    axs[0].set_xlabel(r'Phasenverschiebungswinkel in Grad', fontsize=10)
    axs[0].set_ylabel(r'Szenarienanteil in %', fontsize=10)
    axs[1].set_xlabel(r'Phasenverschiebungswinkel in Grad', fontsize=10)


    #ax.legend(fontsize=10, markerscale=5.0)

    #plt.xticks(fontsize = 8)
    #plt.yticks(fontsize = 8)
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    
    axs[0].set_title('Zentrale BLB')
    axs[1].set_title('Q(U)-Kennlinie')

    plt.savefig('verteilung_phi_actor_qu.png', dpi=350)
    plt.savefig('verteilung_phi_actor_qu.svg')

    
    plt.show()
    plt.clf()

        

check_db_length_compatibility(actor_db_length, ref_db_length)
check_gen_s_sum()
voltage_comparison()
load_comparison()
