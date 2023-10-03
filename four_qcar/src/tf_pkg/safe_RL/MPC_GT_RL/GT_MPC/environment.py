import motion_update
import reward

def environment(X_old, car_id, action_id, t_step, params, dist_id, Level_ratio):
    

    # the selected car moves based on the action
    # 0:maintian  1:accelerate  2:decelerate
    
    X_new = motion_update.motion_update(X_old, car_id, action_id, t_step,  params) 

    # compute reward
    R, params = reward.reward(X_new, car_id, action_id, params, dist_id, Level_ratio)   

    return X_new, R