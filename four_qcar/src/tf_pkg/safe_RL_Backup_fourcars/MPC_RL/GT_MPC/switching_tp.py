import math
import numpy as np
from shapely.geometry import Polygon, LineString

def Time2collistion(X_reward, cx, cy, KD_tree_host, KD_tree_others, car_id, id, Delt_dis_host, Delt_dis_others):
    epsilon_1 = 1e-6
    index_vi = int(X_reward[8][car_id])
    index_nvj = int(X_reward[8][id])
    colli_point_index_vi = KD_tree_host.query([cx, cy],1)[1]
    colli_point_index_nvj = KD_tree_others.query([cx, cy],1)[1]
    vel_vi = X_reward[3, car_id]
    vel_nvj = X_reward[3, id]
    t_car_id = abs((int(colli_point_index_vi)-int(index_vi))*Delt_dis_host/(vel_vi+epsilon_1))
    t_nvj = abs((int(colli_point_index_nvj)-int(index_nvj))*Delt_dis_others/(vel_nvj+epsilon_1))
    if index_nvj > colli_point_index_nvj or index_vi > colli_point_index_vi:
        t_car_id = 1e6
        t_nvj = 0
    print('[idx_vi, coli]:{}, [idx_vj, coli]:{}, [vel_i,vel_j]:{}'.format([index_vi,colli_point_index_vi], [index_nvj,colli_point_index_nvj], [vel_vi, vel_nvj]))
    print('[TTC_i, TTC_j]:{}, [host_dis, other_dis]:{}, epsilon:{}'.format([t_car_id, t_nvj], [Delt_dis_host, Delt_dis_others], epsilon_1))
    return t_car_id, t_nvj

def check_path_conflict(ego_path):
    if ego_path is 1:
        conflict_list = [5, 9]
    elif ego_path is 2:
        conflict_list = [6, 10]
    elif ego_path is 3:
        conflict_list = [7, 11]
    elif ego_path is 4:
        conflict_list = [8, 12]
    elif ego_path is 5:
        conflict_list = [1, 6, 8, 9, 11, 12]
    elif ego_path is 6: 
        conflict_list = [2, 5, 7, 9, 10, 12]
    elif ego_path is 7:
        conflict_list = [3, 6, 8, 9, 10, 11]
    elif ego_path is 8:
        conflict_list = [4, 5, 7, 10, 11, 12]
    elif ego_path is 9:
        conflict_list = [1, 5, 6, 7, 10, 12]
    elif ego_path is 10:
        conflict_list = [2, 6, 7, 8, 9, 11]
    elif ego_path is 11:
        conflict_list = [3, 5, 7, 8, 10, 12]
    elif ego_path is 12:
        conflict_list = [4, 5, 6, 8, 9, 11]
    return conflict_list


def get_graph_tp(X_reward, car_id, params, Level_ratio=0):
    epsilon = 1e-6
    l_car = params.l_car # length of car
    w_car = params.w_car # width of car
    
    num_cars = params.num_cars # number of cars

    l_car_safe = params.l_car_safe_fac * l_car*2 # length of safe outer approximation of car
    w_car_safe = params.w_car_safe_fac * w_car*2# width of safe outer approximation of car
    
    Uncertaity  = l_car * params.W_l_car_fac # Uncertainty for aggressive car
    Non_Uncertaity = l_car # Uncertainty for conservative car
    
    collision_ID = [] 
    cross_conflict_ID = []
    longitudinal_conflict_ID = []
    
    Ego_rectangle_safe = Polygon(
            [[X_reward[0,car_id]-l_car_safe/2*math.cos(X_reward[2,car_id])+w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]-l_car_safe/2*math.sin(X_reward[2,car_id])-w_car_safe/2*math.cos(X_reward[2,car_id])],
            [X_reward[0,car_id]-l_car_safe/2*math.cos(X_reward[2,car_id])-w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]-l_car_safe/2*math.sin(X_reward[2,car_id])+w_car_safe/2*math.cos(X_reward[2,car_id])],
            [X_reward[0,car_id]+l_car_safe/2*math.cos(X_reward[2,car_id])-w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]+l_car_safe/2*math.sin(X_reward[2,car_id])+w_car_safe/2*math.cos(X_reward[2,car_id])],
            [X_reward[0,car_id]+l_car_safe/2*math.cos(X_reward[2,car_id])+w_car_safe/2*math.sin(X_reward[2,car_id]),
            X_reward[1,car_id]+l_car_safe/2*math.sin(X_reward[2,car_id])-w_car_safe/2*math.cos(X_reward[2,car_id])]])
    
    l_car_safe = params.l_car_safe_fac * l_car # length of safe outer approximation of car
    w_car_safe = params.w_car_safe_fac * w_car # width of safe outer approximation of car
    for id in range(0,len(X_reward[1,:])):
        if id!=car_id:
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
                path_conflict = check_path_conflict(int(X_reward[7][car_id]))
                if id not in collision_ID and int(X_reward[7][id]) in path_conflict:
                    collision_ID.append(id)
                
    count = 0    
    Heading_traj_host = np.array(params.waypoints[str(int(X_reward[7, car_id]))])
    KD_tree_host = params.KDtrees_12[str(int(X_reward[7, car_id]))]
    Delt_dis_host = math.sqrt((Heading_traj_host[int(len(Heading_traj_host)/2)][0] - Heading_traj_host[int(len(Heading_traj_host)/2)-1][0])**2+
                            (Heading_traj_host[int(len(Heading_traj_host)/2)][1] - Heading_traj_host[int(len(Heading_traj_host)/2)-1][1])**2)
    Num_point_extr_host = int(Uncertaity/Delt_dis_host)
    Num_point_non_host = int(Non_Uncertaity/Delt_dis_host)
    
    for id in range(0, len(X_reward[0,:])): 
        #w_ext = dist_comb[dist_id]  
        if id!=car_id:
            Heading_traj_others = np.array(params.waypoints[str(int(X_reward[7, id]))])
            KD_tree_others = params.KDtrees_12[str(int(X_reward[7, id]))]
            Delt_dis_others = math.sqrt((Heading_traj_others[int(len(Heading_traj_others)/2)][0] - Heading_traj_others[int(len(Heading_traj_others)/2)-1][0])**2+
                                 (Heading_traj_others[int(len(Heading_traj_others)/2)][1] - Heading_traj_others[int(len(Heading_traj_others)/2)-1][1])**2)
            Num_point_extr = int(Uncertaity/Delt_dis_others)
            Num_point_non = int(Non_Uncertaity/Delt_dis_others)
            
            # if X_reward[9, car_id] == 1:
            #     W_curr_others = int(Level_ratio[(car_id)*(num_cars-1) + count][0]*Num_point_extr)+Num_point_non
            #     W_curr_host = int(Level_ratio[(car_id)*(num_cars-1) + count][0]*Num_point_extr_host)+Num_point_non_host
            # elif X_reward[9, car_id] == 0:
            #     W_curr_others = Num_point_non
            #     W_curr_host = Num_point_non_host
            # else:
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
                if id not in cross_conflict_ID:
                    cross_conflict_ID.append(id)
                # try:    
                #     points = look_ahead_host.intersection(look_ahead_others).coords.xy
                #     points = np.array(points)
                #     cx, cy = float("{:.2f}".format(points[0][0])), float("{:.2f}".format(points[1][0]))
                #     print(car_id+1, id+1, cx, cy)
                #     t_car_id, t_id = Time2collistion(X_reward, cx, cy, KD_tree_host, KD_tree_others, car_id, id, Delt_dis_host, Delt_dis_others)
                #     conflict = (1/((t_car_id - t_id)**2+epsilon))
                    
                # except:
                #     pass
    # print(collision_ID, cross_conflict_ID)
    return collision_ID, cross_conflict_ID

