# -*- coding: utf-8 -*-
import reward_sim
import motion_update_DiGraph
import numpy as np

def Environment_Multi_Sim(X_old1, action_id, t_step, params, L_matrix_all, c_id):
    
    # the selected car moves based on the action
    # 1:maintian  2:turn left  3:turn right  4:accelerate  5:decelerate  6:hard brake
    X_new = np.zeros((X_old1.shape[0], X_old1.shape[1]))
    R = np.zeros((len(action_id), 1))
    for step in range(0, 1):
        size_action_cell = len(action_id)
        for car_id in range(0, size_action_cell):
            # if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
            if X_old1[8][car_id] + 15 < X_old1[6][car_id]:
                X_old1 = motion_update_DiGraph.motion_update(X_old1, car_id, action_id[car_id], t_step, params, L_matrix_all)    
        
        X_new=X_old1.copy()   
            
        for car_id in range(0, size_action_cell): 
            # if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
            R[car_id, step], params = reward_sim.reward_sim(X_new, car_id, action_id[car_id], params)     
            
    return X_new, R