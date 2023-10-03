#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Mingfeng
Date: 27/2021
"""
import math
import numpy as np
HORIZON = 0.7

# def check_preceding(X_old, X_new, car_id, path_id, params):
#     x_old_ = X_old[0][car_id]
#     y_old_ = X_old[1][car_id]
#     yaw_old_ = X_old[2][car_id]
    
#     rear_x_old_ = x_old_ - math.cos(yaw_old_) * 0.128
#     rear_y_old_ = y_old_ - math.sin(yaw_old_) * 0.128
#     x_new_ = X_new[0][car_id]
#     y_new_ = X_new[1][car_id]
#     yaw_new_ = X_new[2][car_id] 

#     rear_x_new_ = x_new_ - math.cos(yaw_new_) * 0.128
#     rear_y_new_ = y_new_ - math.sin(yaw_new_) * 0.128
    
#     targetIndex_old = params.KDtrees_12[path_id].query([rear_x_old_, rear_y_old_],1)[1]
#     targetIndex_new = params.KDtrees_12[path_id].query([rear_x_new_, rear_y_new_],1)[1]

#     proceding = targetIndex_new - targetIndex_old  
                                                                
#     return proceding
def check_preceding(X_new, car_id, params):

    #normalize the proceding position
    x_new_ = X_new[0][car_id]
    y_new_ = X_new[1][car_id]
    yaw_new_ = X_new[2][car_id] 

    rear_x_new_ = x_new_ - math.cos(yaw_new_) * 0.128
    rear_y_new_ = y_new_ - math.sin(yaw_new_) * 0.128
    path_id = X_new[7][car_id]
    path_id = str(int(path_id))
    current_index = params.KDtrees_12[path_id].query([rear_x_new_, rear_y_new_],1)[1]

    # progress = current_index/(X_new[6][car_id] - X_new[5][car_id])
    return current_index
    
def calculateTwistCommand(X_old, car_id, params):
    lad = 0.0  # look ahead distance accumulator
    k = 3.5
    
    x = X_old[0][car_id]
    y = X_old[1][car_id]
    yaw = X_old[2][car_id]
    v = X_old[3][car_id]
    rear_x = x - math.cos(yaw) * 0.128
    rear_y = y - math.sin(yaw) * 0.128
    
    ld = k * v
    path_id = X_old[7][car_id]
    path_id = str(int(path_id))
    
    #print("value of vx is", self.robotstate1[2],"value of vx is", self.robotstate1[5], "value of ld is", ld)
    
    targetIndex = params.KDtrees_12[path_id].query([rear_x, rear_y],1)[1]
    current_index = targetIndex
    
    for i in range(current_index, len(params.waypoints[path_id])):
        if ((i + 1) < len(params.waypoints[path_id])):
            this_x = rear_x
            this_y = rear_y
            next_x = params.waypoints[path_id][i+1][0]
            next_y = params.waypoints[path_id][i+1][1]

            lad = lad + math.hypot(next_x - this_x, next_y - this_y)
            HORIZON = ld
            if (lad > HORIZON):
                targetIndex = i + 1
                break
    #print('Next Waypoint Index',targetIndex)
    targetWaypoint = params.waypoints[path_id][targetIndex]
    targetX = targetWaypoint[0]
    targetY = targetWaypoint[1]
    
    currentX = rear_x
    currentY = rear_y
    
    # get angle difference
    alpha = math.atan2(targetY - currentY, targetX - currentX) - yaw
    l = math.sqrt(math.pow(currentX - targetX, 2) +
                    math.pow(currentY - targetY, 2))
    Kdd = 0.25
    # theta = math.atan(2 * 0.256 * math.sin(alpha) / Kdd*ld)
    theta = math.atan(2 * 0.256 * math.sin(alpha) / l)
    return theta

def motion_update(X_old, car_id, action_id, t_step, params, L_matrix_all):
    
    # path = ['3', '4', '1', '2']

    # 0:maintian  1:turn left   2:turn right  3:accelerate  4:decelerate  5:hard brake 6:increased acceleration
    # 7: left lane change 8: right lane change
    
    lr = params.lr
    lf = params.lf
    
    
    thata = calculateTwistCommand(X_old, car_id, params)
    if X_old[8,car_id] >= X_old[6,car_id]:
        action_id == 2
        thata = 0
    else:
        action_id = action_id
    beta = math.atan((lr/(lr+lf))*math.tan(thata))
    
    X_new = X_old.copy()
    AV_fac = 1  # 
    
    # steer_angle = math.pi/250
    # steer_angle_small = math.pi/360
    # steer_angle_big = math.pi/180
    # steer_angle_lane = math.pi/4
    # max_acc = 4
    # max_dec = 5
    nom_acc = params.max_acc
    nom_dec = params.max_dec
    maintain_acc = params.maintain_acc
       
        
    if action_id == 0: #maintain
        X_new[3,car_id] = X_new[3,car_id] + maintain_acc*t_step
        X_new[2,car_id] = X_new[2,car_id] + (X_new[3,car_id]/lr)*math.sin(beta)*t_step
        
    elif action_id == 1: #accelerate
        X_new[3,car_id] = X_new[3,car_id]+nom_acc*t_step
        if X_new[3,car_id]>params.v_max:
            X_new[3,car_id]=params.v_max
        X_new[2,car_id] = X_new[2,car_id] + (X_new[3,car_id]/lr)*math.sin(beta)*t_step
    
    elif action_id == 2: #decelerate
        X_new[3,car_id] = X_new[3, car_id]-nom_dec*t_step
        if X_new[3,car_id]<params.v_min:
            X_new[3,car_id]=params.v_min
        X_new[2, car_id] = X_new[2,car_id] + (X_new[3,car_id]/lr)*math.sin(beta)*t_step
        
    elif action_id == 3: #Hard accelerate
        X_new[3,car_id] = X_new[3,car_id]+2*nom_acc*t_step
        if X_new[3,car_id]>params.v_max:
            X_new[3,car_id]=params.v_max
        X_new[2,car_id] = X_new[2,car_id] + (X_new[3,car_id]/lr)*math.sin(beta)*t_step
        
    elif action_id == 4: #decelerate
        X_new[3,car_id] = X_new[3, car_id]-3*nom_dec*t_step
        if X_new[3,car_id]<params.v_min:
            X_new[3,car_id]=params.v_min
        X_new[2, car_id] = X_new[2,car_id] + (X_new[3,car_id]/lr)*math.sin(beta)*t_step
        
    X_new[0,car_id] = (X_new[0,car_id]+X_new[3,car_id]*math.cos(X_new[2,car_id]+beta)*t_step)
    X_new[1,car_id] = (X_new[1,car_id]+X_new[3,car_id]*math.sin(X_new[2,car_id]+beta)*t_step)
    current_index = check_preceding(X_new, car_id, params)
    X_new[8, car_id] = current_index
    
    return X_new
    

    # elif action_id == 1:
    #     X_new[3,car_id] = X_new[3,car_id]
    #     beta = math.atan((lr/(lr+lf))*math.tan(steer_angle))
    #     X_new[2,car_id] = X_new[2,car_id]+(X_new[3,car_id]/lr)*math.sin(beta)*t_step
 
            
    # elif action_id == 2:
    #     X_new[3,car_id] = X_new[3,car_id]
    #     beta = math.atan((lr/(lr+lf))*math.tan(-steer_angle))
    #     X_new[2,car_id] = X_new[2,car_id]+(X_new[3,car_id]/lr)*math.sin(beta)*t_step
 
    # elif action_id == 3:
    #     X_new[3,car_id] = X_new[3,car_id]+nom_acc*t_step
    #     if X_new[3,car_id]>params.v_max:
    #         X_new[3,car_id]=params.v_max
    #     X_new[2,car_id] = X_new[2, car_id]
    
    # elif action_id == 4:
    #     X_new[3,car_id] = X_new[3, car_id]-nom_dec*t_step
    #     if X_new[3,car_id]<params.v_min:
    #         X_new[3,car_id]=params.v_min
    #     X_new[2, car_id] = X_new[2, car_id]
    
    # elif action_id == 5:
    #     X_new[3, car_id] = X_new[3, car_id]-max_dec*t_step
    #     if X_new[3,car_id]<params.v_min:
    #         X_new[3,car_id] = params.v_min
    #     X_new[2,car_id] = X_new[2,car_id]
        
    # elif action_id == 6:
    #     X_new[3,car_id] = X_new[3,car_id]+max_acc*t_step
    #     if X_new[3,car_id]>params.v_max:
    #         X_new[3,car_id]=params.v_max
    #     X_new[2,car_id] = X_new[2, car_id]
        
        
    # elif action_id == 7:
    #     X_new[3,car_id] = X_new[3,car_id]+nom_acc*t_step
    #     if X_new[3,car_id]>params.v_max:
    #         X_new[3,car_id]=params.v_max
    #     beta = math.atan((lr/(lr+lf))*math.tan(steer_angle_big))
    #     X_new[2,car_id] = X_new[2,car_id]+(X_new[3,car_id]/lr)*math.sin(beta)*t_step
 
            
    # elif action_id == 8:
    #     X_new[3,car_id] = X_new[3,car_id]+nom_acc*t_step
    #     if X_new[3,car_id]>params.v_max:
    #         X_new[3,car_id]=params.v_max
    #     beta = math.atan((lr/(lr+lf))*math.tan(-steer_angle_big))
    #     X_new[2,car_id] = X_new[2,car_id]+(X_new[3,car_id]/lr)*math.sin(beta)*t_step
        
        
    
        
        
#    elif action_id == 6:
##        if X_new[4, car_id] != 1:
##            nom_acc = fac*nom_acc
#        X_new[3,car_id] = X_new[3,car_id]+nom_acc*t_step
#        if X_new[3,car_id]>params.v_max:
#            X_new[3,car_id]=params.v_max
#        X_new[2,car_id] = X_new[2,car_id]+steer_angle_lane*t_step
#        X_new[0,car_id] = X_new[0,car_id] + X_new[3,car_id]*math.cos(X_new[2,car_id])*t_step
#        X_new[1,car_id] = (X_new[1,car_id]+X_new[3,car_id]*math.sin(X_new[2,car_id])*t_step)
#    
#    elif action_id == 7:
##        if X_new[4,car_id] != 1:
##            nom_acc = fac*nom_acc
#        X_new[3,car_id] = X_new[3,car_id]+nom_acc*t_step
#        if X_new[3,car_id]>params.v_max:
#            X_new[3,car_id]=params.v_max
#        X_new[2,car_id] = X_new[2,car_id]-steer_angle_lane*t_step
#        X_new[0,car_id] = (X_new[0,car_id]+X_new[3,car_id]*math.cos(X_new[2,car_id])*t_step)
#        X_new[1,car_id] = (X_new[1,car_id]+X_new[3,car_id]*math.sin(X_new[2,car_id])*t_step)
    

    