import math
import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from scipy import ndimage

from PIL import Image
from PIL import ImageChops
import os
debug = False

def plot_sim(X_old, params, step, Level_ratio, fig, counter, path):
    _path = path + 'motions'
    isExist = os.path.exists(_path)
    if not isExist:
        os.makedirs(_path)
    l_car = params.l_car 
    w_car = params.w_car

    car_rect_lw = 0.5
    l_car_safe = params.l_car_safe_fac * l_car
    w_car_safe = params.w_car_safe_fac * w_car
    Delt_l = l_car_safe - l_car
    Delt_w = w_car_safe - w_car
    other_waypoints = params.waypoints_shift
    background = params.inter
    
    Non_Uncertaity = np.array([0, 0]) # Uncertainty for conservative car


    color = ['r', 'b', 'y', 'g', 'm', 'c', 'k', 'gray', 'gray','k']
    # plt.figure(fig.number)
    # # plt.cla()
    # ax = plt.gca()
    fig, ax = plt.subplots(1, 1, figsize=(6,6))

    ego_car_id = 1 # AV id
    surrounding_ls = [_id for _id in range(0, len(X_old[0,:]))]
    surrounding_ls.remove(1)
    
    P_ls = []
    for interact_id in surrounding_ls:
        if interact_id ==0:
            index_num = 0
        else:
            index_num = interact_id-1
        # if interact_id ==0:
        #     index_num = 0
        # elif interact_id ==2:
        #     index_num =1
        # elif interact_id == 3:
        #     index_num =2
        # elif interact_id == 4:
        #     index_num =3
        # else:
        #     print('more than 5 cars; plot sim')
        P0 = Level_ratio[ego_car_id * (params.num_cars-1) + index_num, 0]
        P_ls.append(P0)
    # print('car 2: {}, car3: {}, car4: {}'.format(P_ls[0], P_ls[1], P_ls[2]))
    count = 0
    for id in range(0,len(X_old[0,:])):
  
        l_car = params.l_car 
        w_car = params.w_car
        l_car_safe = params.l_car_safe_fac * l_car
        w_car_safe = params.w_car_safe_fac * w_car
        Delt_l = l_car_safe - l_car
        Delt_w = w_car_safe - w_car
        rect = np.array(
            [[X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car / 2 * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car / 2 * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car / 2 * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car / 2 * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + (l_car / 2 ) * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + (l_car / 2 ) * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + (l_car / 2 ) * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + (l_car / 2 ) * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])]])


        coll_rect = np.array(
            [[X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car / 2 * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car / 2 * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car / 2 * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car / 2 * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])]])


        if id != 1:
            l_car_safe = l_car+0.5*Delt_l + 0.5*Delt_l*P_ls[surrounding_ls.index(id)]
            w_car_safe = w_car+0.5*Delt_w + 0.5*Delt_w*P_ls[surrounding_ls.index(id)]
            rectangle_safe = np.array(
                [[X_old[0, id] - l_car_safe / 2 * math.cos(X_old[2, id]) + w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] - l_car_safe / 2 * math.sin(X_old[2, id]) - w_car_safe / 2 * math.cos(X_old[2, id])],
                [X_old[0, id] - l_car_safe / 2 * math.cos(X_old[2, id]) - w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] - l_car_safe / 2 * math.sin(X_old[2, id]) + w_car_safe / 2 * math.cos(X_old[2, id])],
                [X_old[0, id] + l_car_safe / 2 * math.cos(X_old[2, id]) - w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] + l_car_safe / 2 * math.sin(X_old[2, id]) + w_car_safe / 2 * math.cos(X_old[2, id])],
                [X_old[0, id] + l_car_safe / 2 * math.cos(X_old[2, id]) + w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] + l_car_safe / 2 * math.sin(X_old[2, id]) - w_car_safe / 2 * math.cos(X_old[2, id])]])
        else:
            if debug ==True:
                l_car_safe = l_car_safe*1.5
                w_car_safe = w_car_safe*2.5
                l_car =l_car_safe
            rectangle_safe = np.array(
                [[X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) + w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) - w_car_safe / 2 * math.cos(X_old[2, id])],
                [X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) - w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) + w_car_safe / 2 * math.cos(X_old[2, id])],
                [X_old[0, id] + l_car_safe / 2 * math.cos(X_old[2, id]) - w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] + l_car_safe / 2 * math.sin(X_old[2, id]) + w_car_safe / 2 * math.cos(X_old[2, id])],
                [X_old[0, id] + l_car_safe / 2 * math.cos(X_old[2, id]) + w_car_safe / 2 * math.sin(X_old[2, id]),
                X_old[1, id] + l_car_safe / 2 * math.sin(X_old[2, id]) - w_car_safe / 2 * math.cos(X_old[2, id])]])
    


        # plot the disturbance set from the perspective of the AV
        # dist_comb = params.dist_comb
        # w_ext = dist_comb[3]    # choosing [1 1]
        # W_curr = w_ext * np.array([l_car / params.W_l_car_fac, w_car / params.W_w_car_fac])
        Uncertaity  = l_car * params.W_l_car_fac # Uncertainty for aggressive car
        Uncertaity_tp  = l_car * params.W_l_car_fac # Uncertainty for aggressive car
        Non_Uncertaity =1.5* l_car # Uncertainty for conservative car
            # count = 0
            # for car_id in range(0, params.num_cars):
            #     if id != car_id:
            #         for i in range(0, 2):
            #             X_old[i, car_id] = X_old[i, car_id] + W_curr[i] * Level_ratio[(id) * (params.num_cars - 1) + count][0]
            #         count += 1
        
                
        Heading_traj = params.waypoints[str(int(X_old[7, id]))]
        Delt_dis = math.sqrt((Heading_traj[int(len(Heading_traj)/2)][0] - Heading_traj[int(len(Heading_traj)/2)-1][0])**2+(Heading_traj[int(len(Heading_traj)/2)][1] - Heading_traj[int(len(Heading_traj)/2)-1][1])**2)
        
        Num_point_extr = int(Uncertaity/Delt_dis)
        Num_point_non = int(Non_Uncertaity/Delt_dis)
        
        Num_point_extr_tp = int(Uncertaity_tp/Delt_dis)
        if X_old[9, id] == 1:
            W_curr_ls = []
            for p in P_ls:
                W_curr = int(p*Num_point_extr)+Num_point_non
                W_curr_ls.append(W_curr)
        elif X_old[9, id] == 0:
            W_curr = Num_point_non
        else:
            W_curr = Num_point_extr + Num_point_non
        lookahead_tp = Num_point_non +Num_point_extr_tp
        

        # dist_rect = np.array(
        #     [[X_old[0, id] - (l_car_safe) / 2 * math.cos(X_old[2, id]) + (w_car_safe) / 2 * math.sin(X_old[2, id]),
        #       X_old[1, id] - (l_car_safe) / 2 * math.sin(X_old[2, id]) - (w_car_safe) / 2 * math.cos(X_old[2, id])],
        #      [X_old[0, id] - (l_car_safe) / 2 * math.cos(X_old[2, id]) - (w_car_safe) / 2 * math.sin(X_old[2, id]),
        #       X_old[1, id] - (l_car_safe) / 2 * math.sin(X_old[2, id]) + (w_car_safe) / 2 * math.cos(X_old[2, id])],
        #      [X_old[0, id] + (W_curr[0]) / 2 * math.cos(X_old[2, id]) - (W_curr[1]) / 2 * math.sin(X_old[2, id]),
        #       X_old[1, id] + (W_curr[0]) / 2 * math.sin(X_old[2, id]) + (W_curr[1]) / 2 * math.cos(X_old[2, id])],
        #      [X_old[0, id] + (W_curr[0]) / 2 * math.cos(X_old[2, id]) + (W_curr[1]) / 2 * math.sin(X_old[2, id]),
        #       X_old[1, id] + (W_curr[0]) / 2 * math.sin(X_old[2, id]) - (W_curr[1]) / 2 * math.cos(X_old[2, id])]])

        # 
        
        
        if id == ego_car_id:
            color_id = 0
            # Disturbance rectangle
            Heading_traj_0 = other_waypoints['ll'][str(int(X_old[7, id]))]
            wps_tp = np.array(Heading_traj[int(X_old[8, id]):int(X_old[8, id])+lookahead_tp])
            plt.plot(-wps_tp[:, 1], wps_tp[:, 0], color=color[0], LineWidth=car_rect_lw+1, linestyle=':')
            
            wps_0 = np.array(Heading_traj_0[int(X_old[8, id]):int(X_old[8, id])+W_curr_ls[0]])               
            plt.plot(-wps_0[:, 1], wps_0[:, 0], color=color[1], LineWidth=car_rect_lw+1)
            
            wps_2 = np.array(Heading_traj[int(X_old[8, id]):int(X_old[8, id])+W_curr_ls[1]])
            plt.plot(-wps_2[:, 1], wps_2[:, 0], color=color[2], LineWidth=car_rect_lw+1)
            
            Heading_traj_3 = other_waypoints['rr'][str(int(X_old[7, id]))]
            wps_3 = np.array(Heading_traj_3[int(X_old[8, id]):int(X_old[8, id])+W_curr_ls[2]])   
                        
            plt.plot(-wps_3[:, 1], wps_3[:, 0], color=color[3], LineWidth=car_rect_lw+1)
            
            try:
                Heading_traj_4 = other_waypoints['r'][str(int(X_old[7, id]))]
                wps_4 = np.array(Heading_traj_4[int(X_old[8, id]):int(X_old[8, id])+W_curr_ls[3]])
                plt.plot(-wps_4[:, 1], wps_4[:, 0], color=color[4], LineWidth=car_rect_lw+1)
                Heading_traj_5 = other_waypoints['l'][str(int(X_old[7, id]))]
                wps_5 = np.array(Heading_traj_5[int(X_old[8, id]):int(X_old[8, id])+W_curr_ls[5]])
                plt.plot(-wps_5[:, 1], wps_5[:, 0], color=color[6], LineWidth=car_rect_lw+1)
            except:
                pass  
            
            
            plt.plot([-coll_rect[0, 1], -coll_rect[1, 1]], [coll_rect[0, 0], coll_rect[1, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--'), 
            plt.plot([-coll_rect[1, 1], -coll_rect[2, 1]], [coll_rect[1, 0], coll_rect[2, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--'), 
            plt.plot([-coll_rect[2, 1], -coll_rect[3, 1]], [coll_rect[2, 0], coll_rect[3, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--'), 
            plt.plot([-coll_rect[0, 1], -coll_rect[3, 1]], [coll_rect[0, 0], coll_rect[3, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--')
            count += 1
            plt.plot([-rectangle_safe[0, 1], -rectangle_safe[1, 1]], [rectangle_safe[0, 0], rectangle_safe[1, 0]] , color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([-rectangle_safe[1, 1], -rectangle_safe[2, 1]], [rectangle_safe[1, 0], rectangle_safe[2, 0]] , color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([-rectangle_safe[2, 1], -rectangle_safe[3, 1]], [rectangle_safe[2, 0], rectangle_safe[3, 0]] , color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([-rectangle_safe[0, 1], -rectangle_safe[3, 1]], [rectangle_safe[0, 0], rectangle_safe[3, 0]] , color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
        else:
            plt.plot([-rectangle_safe[0, 1], -rectangle_safe[1, 1]], [rectangle_safe[0, 0], rectangle_safe[1, 0]],  color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([-rectangle_safe[1, 1], -rectangle_safe[2, 1]], [rectangle_safe[1, 0], rectangle_safe[2, 0]],  color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([-rectangle_safe[2, 1], -rectangle_safe[3, 1]], [rectangle_safe[2, 0], rectangle_safe[3, 0]],  color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([-rectangle_safe[0, 1], -rectangle_safe[3, 1]], [rectangle_safe[0, 0], rectangle_safe[3, 0]],  color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            
            wps_tp = np.array(Heading_traj[int(X_old[8, id]):int(X_old[8, id])+lookahead_tp])
            plt.plot(-wps_tp[:, 1], wps_tp[:, 0], color=color[-1], LineWidth=car_rect_lw+1, linestyle=':')
            
            wps = np.array(Heading_traj[int(X_old[8, id]):int(X_old[8, id])+W_curr])
                
            plt.plot(-wps[:, 1], wps[:, 0], color=color[id], LineWidth=car_rect_lw+1)
            plt.plot([-coll_rect[0, 1], -coll_rect[1, 1]], [coll_rect[0, 0], coll_rect[1, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--')
            plt.plot([-coll_rect[1, 1], -coll_rect[2, 1]], [coll_rect[1, 0], coll_rect[2, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--')
            plt.plot([-coll_rect[2, 1], -coll_rect[3, 1]], [coll_rect[2, 0], coll_rect[3, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--')
            plt.plot([-coll_rect[0, 1], -coll_rect[3, 1]], [coll_rect[0, 0], coll_rect[3, 0]], color=color[-1], LineWidth=car_rect_lw+1,
                     linestyle='--')
    # intersection scenario
        c_vertical = background[str(1)].tolist().index([0.0, -1.1])

    for i in range(1, 11):
        _data = np.array(background[str(i)])
        if i<3:
            plt.plot(_data[0:c_vertical, 0], _data[0:c_vertical, 1], _data[-c_vertical:-1, 0], _data[-c_vertical:-1, 1], color = 'silver', linestyle = 'dashed')
        elif i >6:
            plt.plot(_data[0:c_vertical, 0], _data[0:c_vertical, 1], color = 'gray', linestyle = 'dashed')
        else:
            plt.plot(_data[:, 0], _data[:, 1], 'k-' ,LineWidth=car_rect_lw+1)
            
    #fig = plt.figure()
    # Setting axes limts
    label_font_size = 20
    limit_size = 2.0
    x_lim = np.array([-limit_size, limit_size])
    y_lim = np.array([-limit_size, limit_size])
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    ax.tick_params(labelsize=label_font_size)

    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(0.5)

    
    # Display Car_id
    display = True
    if display:
        
        move= 0.3
        
        for id in range(0, len(X_old[0,:])):
            if id ==0:
                ax.annotate('Car 1', xy=(-X_old[1, id], X_old[0, id]-move), size=label_font_size)
            elif id ==1:
                ax.annotate('Ego', xy=(-X_old[1, id], X_old[0, id]-move) ,size=label_font_size)
            elif id ==2:
                ax.annotate('Car 2', xy=(-X_old[1, id], X_old[0, id]-move) ,size=label_font_size)
            elif id ==3:
                ax.annotate('Car 3', xy=(-X_old[1, id], X_old[0, id]-move) ,size=label_font_size)
            elif id ==4:
                ax.annotate('Car 4', xy=(-X_old[1, id], X_old[0, id]-move) ,size=label_font_size)
            elif id ==5:
                ax.annotate('Car 5', xy=(-X_old[1, id], X_old[0, id]-move) ,size=label_font_size)    
            elif id ==6:
                ax.annotate('Car 6', xy=(-X_old[1, id], X_old[0, id]-move) ,size=label_font_size)
        #     #ax.annotate('v='+str(X_old[3,0])+'m/s', xy=(5, -10))    
        #     #ax.annotate('v='+str(X_old[3,1])+'m/s', xy=(5, 10))


        
    # plt.yticks([])
    # plt.xlabel('y (m)')
    # plt.xlabel('x (m)')
    #ax.axis('on')
    outfile = 'Test.mp4',
    plot_fname = 'plot',
    plot_format = '.jpg',
    outdir = 'Images',
    fps = 3,
    # plt.tight_layout()
    #plt.show(block=False)
    # plt.savefig(path + '/' + '.pdf', format="pdf", dpi=1200)
    
    plt.savefig(_path+'/'+str(counter)+'.png', dpi = 100)
    # plt.savefig('/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Motion/'+str(counter)+'.png', dpi = 100, bbox_inches='tight', pad_inches=0.5)
    ax.cla()


