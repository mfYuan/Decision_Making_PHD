import itertools
import random
import numpy as np

import csv
import matplotlib.pyplot as plt

# def select_path():
#     path_ls_car1 = {'right':'4', 'strait':'5', 'left':'10'}
#     path_ls_car2 = {'right':'3', 'strait':'8', 'left':'9'}
#     path_ls_car3 = {'right':'1', 'strait':'6', 'left':'11'}
#     path_ls_car4 = {'right':'2', 'strait':'7', 'left':'12'}
#     car1_path = random.choice(path_ls_car1.items())
#     car2_path = random.choice(path_ls_car2.items())
#     car3_path = random.choice(path_ls_car3.items())
#     car4_path = random.choice(path_ls_car4.items()) 
#     print('car1:{}, car2:{}, car3:{}, car4:{}'.format(car1_path, car2_path, car3_path, car4_path))
#     return [car1_path, car2_path, car3_path, car4_path]

# select_path()

# print(list(itertools.product([-1, 1], repeat=2)))
######
# folder_path = '/home/sdcnlab025/ROS_test/four_qcar/src/waypoint_loader/waypoints/waypoints12.csv'
# pos_list = []
# with open(folder_path, 'r') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         pos_list.append(list(map(float, row)))
#     pos_list = np.array(pos_list)
#     print(pos_list.shape)
#     plt.plot(pos_list[:,1], pos_list[:,0])
#     plt.show()
######
# folder_path = '/home/sdcnlab025/ROS_test/four_qcar/src/waypoint_loader/waypoints/'
# path = []
# for i in range(1, 13):
#     file_name = folder_path + 'waypoints'+str(i) + '.csv'
#     pos_list = []
#     with open(file_name, 'r') as csvfile:
#         spamreader = csv.reader(csvfile, delimiter=',')
#         for row in spamreader:
#             pos_list.append(list(map(float, row)))
#         pos_list = np.array(pos_list)
#     path.append(pos_list)
# path = np.array(path)

# for i in range(0, 12):
#     if i%2 == 0:
#         plt.plot(path[i][:, 0], path[i][:, 1], '-.', label='path'+str(i+1), linewidth=5)
#     else:
#         plt.plot(path[i][:, 0], path[i][:, 1], label='path'+str(i+1), linewidth=5)
#     plt.legend()
# plt.savefig('/home/sdcnlab025/ROS_test/four_qcar/src/waypoint_loader/waypoints/'+ 'intersection'+'.png')
# plt.show()
Q_init = -1e6 # initial Q value
Q_value_2 = [[i]  for i in range(3)]
print(Q_value_2[0])




# for id_2 in range(0, 3):
#     if Q_value_2[id_2][0] != Q_init:
#         Q_value_2[id_2] = list([0])
#     else:
#         Q_value_2[id_2] = Q_value_2[id_2] + list([1])
# print([1]+[1])

