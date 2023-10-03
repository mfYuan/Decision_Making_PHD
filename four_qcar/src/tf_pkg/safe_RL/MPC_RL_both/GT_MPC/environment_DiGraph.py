import motion_update_DiGraph
import reward_DiGraph

def environment(X_old, car_id, action_id, t_step, params, dist_id, Level_ratio, L_matrix_all):
    # the selected car moves based on the action
    # 0:maintian  1:accelerate  2:decelerate
    X_new = motion_update_DiGraph.motion_update(X_old, car_id, action_id, t_step,  params, L_matrix_all) 
    # compute reward
    R, params = reward_DiGraph.reward(X_new, car_id, action_id, params, dist_id, Level_ratio, L_matrix_all)   
    return X_new, R