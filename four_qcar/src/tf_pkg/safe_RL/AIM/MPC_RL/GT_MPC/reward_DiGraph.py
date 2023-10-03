import math
import numpy as np
from shapely.geometry import Polygon, LineString

def Time2collistion(X_reward, cx, cy, KD_tree_host, KD_tree_others, car_id, id, Delt_dis_host, Delt_dis_others):
    epsilon_1 = 1e-6
    index_vi = int(X_reward[8][car_id])#-int(0.1/Delt_dis_host)
    index_nvj = int(X_reward[8][id])#-int(0.1/Delt_dis_others)
    colli_point_index_vi = KD_tree_host.query([cx, cy],1)[1]
    colli_point_index_nvj = KD_tree_others.query([cx, cy],1)[1]
    vel_vi = X_reward[3, car_id]
    vel_nvj = X_reward[3, id]
    t_car_id = abs((int(colli_point_index_vi)-int(index_vi))*Delt_dis_host/(vel_vi+epsilon_1))
    t_nvj = abs((int(colli_point_index_nvj)-int(index_nvj))*Delt_dis_others/(vel_nvj+epsilon_1))
    if index_nvj > colli_point_index_nvj or index_vi > colli_point_index_vi:
        t_car_id = 1e6
        t_nvj = 0
    if vel_vi < 0.005 or vel_nvj< 0.005:
        flag = 'off'
    else:
        flag = 'on'
        
    # print('[idx_vi, coli]:{}, [idx_vj, coli]:{}, [vel_i,vel_j]:{}'.format([index_vi,colli_point_index_vi], [index_nvj,colli_point_index_nvj], [vel_vi, vel_nvj]))
    # print('[TTC_i, TTC_j]:{}, [host_dis, other_dis]:{}, epsilon:{}'.format([t_car_id, t_nvj], [Delt_dis_host, Delt_dis_others], epsilon_1))
    return t_car_id, t_nvj, flag


def reward(X_reward, car_id, action_id, params, dist_id, Level_ratio, L_matrix_all):
    episode = params.episode
    complete_flag = params.complete_flag
    epsilon = 1e-6
    l_car = params.l_car # length of car
    w_car = params.w_car # width of car
    
    num_cars = params.num_cars # number of cars
    v_ref = params.v_nominal # reference velocity
  
    l_car_safe = params.l_car_safe_fac * l_car # length of safe outer approximation of car
    w_car_safe = params.w_car_safe_fac * w_car # width of safe outer approximation of car
    
    Delt_l = l_car_safe - l_car
    Delt_w = w_car_safe - w_car
    
    Uncertaity  = l_car * params.W_l_car_fac # Uncertainty for aggressive car
    Non_Uncertaity = 1.5*l_car # Uncertainty for conservative car
    
    
    Safe = 0
    Safe_Penalty = -1e4  
    # Speed reward
    Speed = 0.0
    
    if car_id ==1:
        _surrounding_ls = [_id for _id in range(0, len(X_reward[0,:]))]
        _surrounding_ls.remove(1)
        P_ls = []
        for interact_id in _surrounding_ls:
            if interact_id ==0:
                _index_num = 0
            else:
                _index_num = interact_id-1
            # elif interact_id ==2:
            #     index_num =1
            # elif interact_id == 3:
            #     index_num =2
            # elif interact_id == 4:
            #     index_num =3
            # else:
            #     print('more than 5 cars; plot sim')
            P0 = Level_ratio[1 * (params.num_cars-1) + _index_num, 0]
            P_ls.append(P0)
    
    Ego_rectangle_safe = Polygon(
            [[X_reward[0,car_id]-l_car_safe/2*math.cos(X_reward[2,car_id])+w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]-l_car_safe/2*math.sin(X_reward[2,car_id])-w_car_safe/2*math.cos(X_reward[2,car_id])],
            [X_reward[0,car_id]-l_car_safe/2*math.cos(X_reward[2,car_id])-w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]-l_car_safe/2*math.sin(X_reward[2,car_id])+w_car_safe/2*math.cos(X_reward[2,car_id])],
            [X_reward[0,car_id]+l_car_safe/2*math.cos(X_reward[2,car_id])-w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]+l_car_safe/2*math.sin(X_reward[2,car_id])+w_car_safe/2*math.cos(X_reward[2,car_id])],
            [X_reward[0,car_id]+l_car_safe/2*math.cos(X_reward[2,car_id])+w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]+l_car_safe/2*math.sin(X_reward[2,car_id])-w_car_safe/2*math.cos(X_reward[2,car_id])]])
    
    for id in range(0,len(X_reward[1,:])):
        if id!=car_id and L_matrix_all[car_id][id] == -1:
            if car_id == 1:
                l_car_safe = l_car+0.5*Delt_l + 0.5*Delt_l*P_ls[_surrounding_ls.index(id)]
                w_car_safe = w_car+0.5*Delt_w + 0.5*Delt_w*P_ls[_surrounding_ls.index(id)]
            Other_rectangle_safe = Polygon(
                [[X_reward[0,id]-l_car_safe/2*math.cos(X_reward[2,id])+w_car_safe/2*math.sin(X_reward[2,id]),
                  X_reward[1,id]-l_car_safe/2*math.sin(X_reward[2,id])-w_car_safe/2*math.cos(X_reward[2,id])],
                [X_reward[0,id]-l_car_safe/2*math.cos(X_reward[2,id])-w_car_safe/2*math.sin(X_reward[2,id]),
                 X_reward[1,id]-l_car_safe/2*math.sin(X_reward[2,id])+w_car_safe/2*math.cos(X_reward[2,id])],
                [X_reward[0,id]+l_car_safe/2*math.cos(X_reward[2,id])-w_car_safe/2*math.sin(X_reward[2,id]),
                 X_reward[1,id]+l_car_safe/2*math.sin(X_reward[2,id])+w_car_safe/2*math.cos(X_reward[2,id])],
                [X_reward[0,id]+l_car_safe/2*math.cos(X_reward[2,id])+w_car_safe/2*math.sin(X_reward[2,id]),
                 X_reward[1,id]+l_car_safe/2*math.sin(X_reward[2,id])-w_car_safe/2*math.cos(X_reward[2,id])]])

            if Ego_rectangle_safe.intersects(Other_rectangle_safe):
                Safe = Safe + Safe_Penalty
                
            else:
                Speed = -1e3*abs(X_reward[3, car_id] - v_ref)
                
                
    # Speed reward
    
                
    conflict = 0
    conflict_Penalty = -1e4
    count = 0    
    Heading_traj_host = np.array(params.waypoints[str(int(X_reward[7, car_id]))])
    KD_tree_host = params.KDtrees_12[str(int(X_reward[7, car_id]))]
    Delt_dis_host = math.sqrt((Heading_traj_host[int(len(Heading_traj_host)/2)][0] - Heading_traj_host[int(len(Heading_traj_host)/2)-1][0])**2+
                            (Heading_traj_host[int(len(Heading_traj_host)/2)][1] - Heading_traj_host[int(len(Heading_traj_host)/2)-1][1])**2)
    Num_point_extr_host = int(Uncertaity/Delt_dis_host)
    Num_point_non_host = int(Non_Uncertaity/Delt_dis_host)
    
    for id in range(0, len(X_reward[0,:])):
        dist_comb = params.dist_comb # distance combination 
        #w_ext = dist_comb[dist_id]  
        if id!=car_id and L_matrix_all[car_id][id] == -1:
            Heading_traj_others = np.array(params.waypoints[str(int(X_reward[7, id]))])
            KD_tree_others = params.KDtrees_12[str(int(X_reward[7, id]))]
            Delt_dis_others = math.sqrt((Heading_traj_others[int(len(Heading_traj_others)/2)][0] - Heading_traj_others[int(len(Heading_traj_others)/2)-1][0])**2+
                                 (Heading_traj_others[int(len(Heading_traj_others)/2)][1] - Heading_traj_others[int(len(Heading_traj_others)/2)-1][1])**2)
            Num_point_extr = int(Uncertaity/Delt_dis_others)
            Num_point_non = int(Non_Uncertaity/Delt_dis_others)
            
            if X_reward[9, car_id] == 1:
                if id ==0:
                    index_num =0
                elif id >1:
                    index_num =id - 1
                
                W_curr_others = int(Level_ratio[(car_id)*(num_cars-1) + index_num][0]*Num_point_extr)+Num_point_non
                W_curr_host = int(Level_ratio[(car_id)*(num_cars-1) + index_num][0]*Num_point_extr_host)+Num_point_non_host
            elif X_reward[9, car_id] == 0:
                W_curr_others = Num_point_non
                W_curr_host = Num_point_non_host
            else:
                W_curr_others = Num_point_extr + Num_point_non
                W_curr_host = Num_point_extr_host + Num_point_non_host
            
            if int(X_reward[8, car_id])+W_curr_host<len(Heading_traj_host):
                look_ahead_host = LineString(Heading_traj_host[int(X_reward[8, car_id]):int(X_reward[8, car_id])+W_curr_host])
            else:
                look_ahead_host = LineString(Heading_traj_host[len(Heading_traj_host)-4:len(Heading_traj_host)-1])
            if int(X_reward[8, id])+W_curr_others<len(Heading_traj_others):
                look_ahead_others = LineString(Heading_traj_others[int(X_reward[8, id]):int(X_reward[8, id])+W_curr_others])
            else:
                look_ahead_others = LineString(Heading_traj_others[len(Heading_traj_others)-4:len(Heading_traj_others)-1])
            count = count + 1
            
            if look_ahead_host.intersects(look_ahead_others):
                #conflict = conflict + conflict_Penalty
                # print('Yes', car_id, id)
                
                try:    
                    points = look_ahead_host.intersection(look_ahead_others).coords.xy
                    points = np.array(points)
                    cx, cy = float("{:.2f}".format(points[0][0])), float("{:.2f}".format(points[1][0]))
                    # print(car_id+1, id+1, cx, cy)
                    t_car_id, t_id, status = Time2collistion(X_reward, cx, cy, KD_tree_host, KD_tree_others, car_id, id, Delt_dis_host, Delt_dis_others)
                    conflict = conflict + (1/((t_car_id - t_id)**2+epsilon))*conflict_Penalty
                    Speed = 0
                    # conflict = conflict*conflict_Penalty
                    # if status == 'off':
                    #     conflict = 0
                    # if car_id ==1:
                    #     print('cross', conflict)
                except:
                    pass

    # Collision penalty
    Colli = 0
    Colli_Penalty = -1e6
    # ???!!!
    l_car_safe_front = 1 * l_car
    l_car_safe_back = 1 * l_car
    w_car_safe = 1 * w_car


    Ego_rectangle = Polygon(
        [[X_reward[0,car_id]-l_car_safe_back/2*math.cos(X_reward[2,car_id])+w_car_safe/2*math.sin(X_reward[2,car_id]),
          X_reward[1,car_id]-l_car_safe_back/2*math.sin(X_reward[2,car_id])-w_car_safe/2*math.cos(X_reward[2,car_id])],
        [X_reward[0,car_id]-l_car_safe_back/2*math.cos(X_reward[2,car_id])-w_car_safe/2*math.sin(X_reward[2,car_id]),
         X_reward[1,car_id]-l_car_safe_back/2*math.sin(X_reward[2,car_id])+w_car_safe/2*math.cos(X_reward[2,car_id])],
        [X_reward[0,car_id]+l_car_safe_front/2*math.cos(X_reward[2,car_id])-w_car_safe/2*math.sin(X_reward[2,car_id]),
         X_reward[1,car_id]+l_car_safe_front/2*math.sin(X_reward[2,car_id])+w_car_safe/2*math.cos(X_reward[2,car_id])],
        [X_reward[0,car_id]+l_car_safe_front/2*math.cos(X_reward[2,car_id])+w_car_safe/2*math.sin(X_reward[2,car_id]),
         X_reward[1,car_id]+l_car_safe_front/2*math.sin(X_reward[2,car_id])-w_car_safe/2*math.cos(X_reward[2,car_id])]])
    
    for id in range(0,len(X_reward[1,:])):
        if id!=car_id and L_matrix_all[car_id][id] == -1:
            Other_rectangle = Polygon(
                [[X_reward[0,id]-l_car_safe_back/2*math.cos(X_reward[2,id])+w_car_safe/2*math.sin(X_reward[2,id]),
                  X_reward[1,id]-l_car_safe_back/2*math.sin(X_reward[2,id])-w_car_safe/2*math.cos(X_reward[2,id])],
                [X_reward[0,id]-l_car_safe_back/2*math.cos(X_reward[2,id])-w_car_safe/2*math.sin(X_reward[2,id]),
                 X_reward[1,id]-l_car_safe_back/2*math.sin(X_reward[2,id])+w_car_safe/2*math.cos(X_reward[2,id])],
                [X_reward[0,id]+l_car_safe_front/2*math.cos(X_reward[2,id])-w_car_safe/2*math.sin(X_reward[2,id]),
                 X_reward[1,id]+l_car_safe_front/2*math.sin(X_reward[2,id])+w_car_safe/2*math.cos(X_reward[2,id])],
                [X_reward[0,id]+l_car_safe_front/2*math.cos(X_reward[2,id])+w_car_safe/2*math.sin(X_reward[2,id]),
                 X_reward[1,id]+l_car_safe_front/2*math.sin(X_reward[2,id])-w_car_safe/2*math.cos(X_reward[2,id])]])

            if Ego_rectangle.intersects(Other_rectangle):
                Colli = Colli + Colli_Penalty
                
                
                

    # Lane overlap violation penalty
    # eps = w_car/8
    # Lane_1 = Polygon([[0, w_lane-eps],[0, w_lane+eps],[l_road, w_lane+eps],[l_road, w_lane-eps]])
    # Lane_2 = Polygon([[0, 2*w_lane-eps],[0, 2*w_lane+eps],[l_road, 2*w_lane+eps],[l_road, 2*w_lane-eps]])
    
    # Ego_rectangle = Polygon(
    #     [[X_reward[0,car_id]-l_car/2*math.cos(X_reward[2,car_id])+w_car/2*math.sin(X_reward[2,car_id]),
    #       X_reward[1,car_id]-l_car/2*math.sin(X_reward[2,car_id])-w_car/2*math.cos(X_reward[2,car_id])],
    #     [X_reward[0,car_id]-l_car/2*math.cos(X_reward[2,car_id])-w_car/2*math.sin(X_reward[2,car_id]),
    #      X_reward[1,car_id]-l_car/2*math.sin(X_reward[2,car_id])+w_car/2*math.cos(X_reward[2,car_id])],
    #     [X_reward[0,car_id]+l_car/2*math.cos(X_reward[2,car_id])-w_car/2*math.sin(X_reward[2,car_id]),
    #      X_reward[1,car_id]+l_car/2*math.sin(X_reward[2,car_id])+w_car/2*math.cos(X_reward[2,car_id])],
    #     [X_reward[0,car_id]+l_car/2*math.cos(X_reward[2,car_id])+w_car/2*math.sin(X_reward[2,car_id]),
    #      X_reward[1,car_id]+l_car/2*math.sin(X_reward[2,car_id])-w_car/2*math.cos(X_reward[2,car_id])]])
    
    
    # Lane_overlap = 0
    # Lane_overlap_penalty = -1e4
    # if abs(X_reward[2, car_id]- X_reward[7,car_id] ) <= 1e-2:
    #     if Ego_rectangle.intersects(Lane_1) or Ego_rectangle.intersects(Lane_2):
    #         Lane_overlap = Lane_overlap + Lane_overlap_penalty    
            
            

    
    # Lane off-center penalty
    # Lane = 0
    # LC_penalty = -1e-5/(w_lane/2)
    
    # for i in range(0, num_lanes):
    #     if X_reward[1,car_id] <= (i+1)*w_lane and X_reward[1,car_id] > (i)*w_lane:
    #         lane = i+1
    #         break  
    #     elif X_reward[1,car_id] > num_lanes*w_lane:
    #         lane = num_lanes
    #     elif X_reward[1,car_id] < 0:
    #         lane = 1
    # lane_center = (lane-1)*w_lane + (w_lane/2)     
    # Lane = Lane + LC_penalty*abs(lane_center-(X_reward[1,car_id]))
    

    # Completion reward
    Complete = 0
    Complete_Penalty = -1e4
    
    if X_reward[8,car_id] < X_reward[6,car_id]:
        Complete = Complete +  Complete_Penalty*abs(X_reward[8,car_id] - X_reward[6,car_id])/abs(X_reward[6,car_id]-X_reward[5,car_id])
    else:
        complete_flag[episode, car_id] = 1
    

    # if X_reward[8,car_id]<X_reward[5,car_id]:
    #     # if X_reward[4,car_id] == 0: # if not AV
    #         # x pos error
    #         Complete = Complete + 1e-5 * Complete_Penalty/X_reward[5,car_id]*abs(X_reward[5,car_id]-X_reward[0,car_id])
    #         # orientation error
    #         Complete = Complete + 1e-3 * Complete_Penalty*abs(math.sin(X_reward[2,car_id] - X_reward[7,car_id] ))

    #     # elif abs(X_reward[6,car_id]-(X_reward[1,car_id])) <= 1e-0:  # if not AV and y error is within threshold
    #     #
    #     #     Complete = Complete + 1e-5 * Complete_Penalty/X_reward[5,car_id]*abs(X_reward[5,car_id]-X_reward[0,car_id])
    #     #     Complete = Complete + 1e-3 * Complete_Penalty*abs(math.sin(X_reward[2,car_id] - X_reward[7,car_id] ))
            
    #         Complete = Complete + 1e4 * Complete_Penalty/X_reward[6,car_id]*abs(X_reward[6,car_id]-(X_reward[1,car_id]))
    # else:
    #     complete_flag[episode,car_id] = 1
        
 

    # if lane == num_lanes:
    #     Speed = -1e2*abs(X_reward[3, car_id] - v_ref*1.05)
    # else:
    #     Speed = -1e2 * abs(X_reward[3, car_id] - v_ref)


    

    # # Effort penalty
    # Effort = 0
    # #Effort_penalty = -1e1
    # if(action_id==1 or action_id==2):
    #     Effort = 0
    # else:
    #     Effort = 0
        
        

    # # Encourage braking if opponent vehicle is parallel
    # Brake = 0
    # Brake_reward = 1e4
    # if X_reward[4,car_id] == 1:
    #     des_lane = math.ceil(X_reward[6,car_id]/w_lane)
    #     for id in range(0,len(X_reward[1,:])):
    #         if id!=car_id:
    #             opp_lane = math.ceil(X_reward[1,id]/w_lane)
    #             if opp_lane == des_lane:
    #                 if (X_reward[0,id]-5*l_car <= X_reward[0,car_id] <= X_reward[0,id]+0.25*l_car):
    #                     Brake = Brake + +Brake_reward * (1/X_reward[3, car_id])




    # # Local Reward
    # R_l = (Off_road*0 + Colli  + Safe*0   + Complete + Speed*0 + Effort*0  + Lane *0  + Lane_overlap *0 + Brake*0)
    # if X_reward[4,car_id] == 1 and not(abs(X_reward[6,car_id]-(X_reward[1,car_id])) <= w_lane/2):
    #     R_l = (Off_road*0 + Colli + Safe*0  + Complete*0 + Speed * 1e-2+ Effort *0+ Lane *0 + Lane_overlap*0 + Brake*0)

    # params.complete_flag = complete_flag
    # Local Reward
    R_l = (Colli + Safe + conflict + Complete + Speed)
    # if X_reward[4,car_id] == 1 and not(abs(X_reward[6,car_id]-(X_reward[1,car_id])) <= w_lane/2):
    #     R_l = (Off_road*0 + Colli + Safe*0  + Complete*0 + Speed * 1e-2)

    params.complete_flag = complete_flag
    R = R_l 
    return R, params


#     ### Social Reward 
    
#     if X_reward[4, car_id]==1:
#         # Speed
#         v_target = params.v_target
#         v_sum = 0

#         for i in range(0, num_cars):
#             if X_reward[4, i] == 1:
#                 v_sum = v_sum + X_reward[3, i]    

#         v_avg = v_sum / params.num_AV
    
#         Vavg_penalty = -1e-1
#         R_Vavg = Vavg_penalty*abs(v_avg-v_target)
    
    
    
#         #Headway
#         num_AV_lane_1 = 0 
#         num_AV_lane_2 = 0 
#         num_AV_lane_3 = 0
    
#         AV_x_lane_1 = np.empty(shape=[1, 0])
#         AV_x_lane_2 = np.empty(shape=[1, 0])
#         AV_x_lane_3 = np.empty(shape=[1, 0])
    
#         for i in range(0, num_cars):
#             if X_reward[4, i] == 1:
#                 if (X_reward[1, i] >= 0) and (X_reward[1, i] <= w_lane):
#                     num_AV_lane_1 = num_AV_lane_1 + 1
#                     #if X_reward[2,i]==0:
#                     AV_x_lane_1 = np.append(AV_x_lane_1, X_reward[0, i])
#                 elif (X_reward[1, i] >= w_lane) and (X_reward[1, i] <= 2*w_lane):
#                     num_AV_lane_2 = num_AV_lane_2 + 1
#                     #if X_reward[2,i]==0:
#                     AV_x_lane_2 = np.append(AV_x_lane_2, X_reward[0, i])              
#                 elif (X_reward[1, i] >= 2*w_lane) and (X_reward[1, i] <= 3*w_lane):
#                     num_AV_lane_3 = num_AV_lane_3 + 1
#                     #if X_reward[2,i]==0:
#                     AV_x_lane_3 = np.append(AV_x_lane_3, X_reward[0, i])
          
    
    
#         h_opt = l_car_safe
#         R_s = 0
#         Prop_headway_penalty = -1e-4
#         Head_way_Penalty = -1e2
#         Head_way_Penalty_tilt = -1e4*0
    
#         if num_AV_lane_1 ==0:
#             head_1 = 0
#             R_s = R_s + 0
#         elif num_AV_lane_1 ==1:
#             head_1 = 0
#             R_s = R_s + Head_way_Penalty
#         else:
#             head_1 = 0
#             head_sum_opt_1 = (num_AV_lane_1 - 1)*h_opt # desired value
#             if len(AV_x_lane_1)>=2:
#                 for j in range(0, num_AV_lane_1-1):
#                     head_temp = AV_x_lane_1[j+1] - AV_x_lane_1[j] - l_car
#                     head_1 = abs(head_temp) + head_1
# #                if head_1 >= head_sum_opt_1:
#                     R_s = R_s + abs(head_sum_opt_1 - head_1)*Prop_headway_penalty
# #                else:
# #                   R_s = R_s + -1e6    
#             else:
#                 R_s = R_s + Head_way_Penalty_tilt


#         if num_AV_lane_2 ==0:
#             head_2 = 0
#             R_s = R_s + 0
#         elif num_AV_lane_2 ==1:
#             head_2 = 0
#             R_s = R_s + Head_way_Penalty
#         else:
#             head_2 = 0
#             head_sum_opt_2 = (num_AV_lane_2-1)*h_opt
#             if len(AV_x_lane_2)>=2:
#                 for j in range(0, num_AV_lane_2-1):
#                     head_temp = AV_x_lane_2[j+1] - AV_x_lane_2[j]-l_car
#                     head_2 = abs(head_temp) + head_2
# #               if head_2 >= head_sum_opt_2:
#                     R_s = R_s + abs(head_sum_opt_2 - head_2) * Prop_headway_penalty
# #               else:
# #                   R_s = R_s + -1e6
#             else:
#                 R_s = R_s + Head_way_Penalty_tilt
    
#         if num_AV_lane_3 == 0:
#             head_3 = 0
#             R_s = R_s + 0
#         elif num_AV_lane_3 == 1:
#             head_3 = 0
#             R_s = R_s + Head_way_Penalty
#         else:
#             head_3 = 0
#             head_sum_opt_3 = (num_AV_lane_3-1)*h_opt
#             if len(AV_x_lane_3)>=2:
#                 for j in range(0, num_AV_lane_3-1):
#                     head_temp = AV_x_lane_3[j+1]-AV_x_lane_3[j]-l_car
#                     head_3 = abs(head_temp) + head_3   
# #               if head_3 >= head_sum_opt_3:
#                     R_s = R_s + abs(head_sum_opt_3 - head_3)*Prop_headway_penalty
# #               else:
# #                   R_s = R_s + -1e6
#             else:
#                 R_s = R_s + Head_way_Penalty_tilt
     
        
#         R_s = R_s + R_Vavg
    
    
#     else:
#         R_s = 0
    
    