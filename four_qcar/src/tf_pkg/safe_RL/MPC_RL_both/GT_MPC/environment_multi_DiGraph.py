# -*- coding: utf-8 -*-
import numpy as np
import motion_update_DiGraph

def environment_multi(X_old, action_id, t_step, params, L_matrix_all, c_id):
    
    X_new = X_old.copy()
    
    # the selected car moves based on the action
    # 1:maintian  2:turn left  3:turn right  4:accelerate  5:decelerate  6:hard brake
    
    X_sub = np.zeros((len(action_id[0]), X_new.shape[0], X_new.shape[1]))#shape: (2, 8, 4)
    
    for step in range(0, len(action_id[0])):
        size_action_cell = len(action_id) #4
        #print("size_action_cell: ", size_action_cell)
        for car_id in range(0, size_action_cell):
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                X_new = motion_update_DiGraph.motion_update(X_new, car_id, action_id[car_id][step], t_step, params, L_matrix_all)
        X_sub[step,:,:]=X_new.copy()     
    
    return X_sub


