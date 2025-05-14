# import os
# # display final results using the logger or matplotlib lowk idek
# from utils import logger 

# def log_gradients_from_file(filename: str, metric_name: str):
#     with open(filename, "r") as f:
#         for line in f:
#             i, g = line.strip().split(",")
#             i = int(i)
#             g = float(g)
#             logger.add_logged_value(name=metric_name, value=g, t=i)

# #particle  number 
# log_gradients_from_file("particle_number_QE.txt", "particle_number_QE")
# log_gradients_from_file("spin_z_QE.txt", "spin_z_QE")
# log_gradients_from_file("total_spin_squared_QE.txt", "total_spin_squared_QE")

# #spin z
# log_gradients_from_file("particle_number_Qubit.txt", "spin_z_Qubit")
# log_gradients_from_file("spin_z__Qubit.txt", "total_spin_squared_Qubit")
# log_gradients_from_file("total_spin_squared_Qubit.txt", "total_spin_squared_Qubit")

# #total spin squared
# log_gradients_from_file("spin_z_combined.txt", "spin_z_combined")
# log_gradients_from_file("total_spin_squared_combined.txt", "total_spin_squared_combined")
# log_gradients_from_file("total_spin_squared_combined.txt", "total_spin_squared_combined")

import matplotlib as plt 
import adapt_vqe 
'''
which molecule(s) should we work with? 
1) H2
2) LiH 
3) BeH2 (perhaps) 

bond lengths for these molecules (varied):
1) H2 - 0.7, 1.4, 2.8 (if not too big)
2) LiH - 2.0, 3.0, 4.0 (use the same bond lengths for BeH2?)

stuff to plot in matplot lib (figure out how to access this information)
expectation value of...
1) particle number 
2) spin z projection
3) total spin squared

do this for all 3 pools (once we have all 3 pools)
1) QE--done
2) Qubit--not done
3) Combined--not done

9 total plots (omit any that are not interesting)

what data analysis are we performing?
'''

#QE Pool

#run with H2, bond length 0.7, store particle number 
# x = iterations, y = particle number 
#python3 -m tj_adapt_vqe

particle_number_ev = []

x = adapt_vqe.max_adapt_iter
y = particle_number_ev
plt.plot(x,y)