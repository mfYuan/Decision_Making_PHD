import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from scipy import ndimage

from PIL import Image
from PIL import ImageChops


def plot_sim(X_old, params, step, Level_ratio, fig):
   
    l_car = params.l_car 
    w_car = params.w_car

    plot_format = params.plot_format
    car_rect_lw = 0.25
    l_car_safe = params.l_car_safe_fac * l_car
    w_car_safe = params.w_car_safe_fac * w_car


    color = ['b','r','m','g', 'k']
    plt.figure(fig.number)
    plt.cla()
    ax = plt.gca()

    count = 0
    for id in range(0,len(X_old[0,:])):
        rect = np.array(
            [[X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] - l_car / 2 * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car / 2 * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car / 2 * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car / 2 * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car / 2 * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car / 2 * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + (l_car / 2 - 1) * math.cos(X_old[2, id]) - w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + (l_car / 2 - 1) * math.sin(X_old[2, id]) + w_car / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + (l_car / 2 - 1) * math.cos(X_old[2, id]) + w_car / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + (l_car / 2 - 1) * math.sin(X_old[2, id]) - w_car / 2 * math.cos(X_old[2, id])]])


        coll_rect = np.array(
            [[X_old[0, id] - l_car_safe / 2 * math.cos(X_old[2, id]) + w_car_safe / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car_safe / 2 * math.sin(X_old[2, id]) - w_car_safe / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] - l_car_safe / 2 * math.cos(X_old[2, id]) - w_car_safe / 2 * math.sin(X_old[2, id]),
              X_old[1, id] - l_car_safe / 2 * math.sin(X_old[2, id]) + w_car_safe / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car_safe / 2 * math.cos(X_old[2, id]) - w_car_safe / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car_safe / 2 * math.sin(X_old[2, id]) + w_car_safe / 2 * math.cos(X_old[2, id])],
             [X_old[0, id] + l_car_safe / 2 * math.cos(X_old[2, id]) + w_car_safe / 2 * math.sin(X_old[2, id]),
              X_old[1, id] + l_car_safe / 2 * math.sin(X_old[2, id]) - w_car_safe / 2 * math.cos(X_old[2, id])]])

        l_car_safe_front = 1.1 * l_car
        l_car_safe_back = 1.1 * l_car
        w_car_safe = 1.1 * w_car


        safe_rect = np.array(
                [[X_old[0,id]-l_car_safe_back/2*math.cos(X_old[2,id])+w_car_safe/2*math.sin(X_old[2,id]),
                  X_old[1,id]-l_car_safe_back/2*math.sin(X_old[2,id])-w_car_safe/2*math.cos(X_old[2,id])],
                [X_old[0,id]-l_car_safe_back/2*math.cos(X_old[2,id])-w_car_safe/2*math.sin(X_old[2,id]),
                 X_old[1,id]-l_car_safe_back/2*math.sin(X_old[2,id])+w_car_safe/2*math.cos(X_old[2,id])],
                [X_old[0,id]+l_car_safe_front/2*math.cos(X_old[2,id])-w_car_safe/2*math.sin(X_old[2,id]),
                 X_old[1,id]+l_car_safe_front/2*math.sin(X_old[2,id])+w_car_safe/2*math.cos(X_old[2,id])],
                [X_old[0,id]+l_car_safe_front/2*math.cos(X_old[2,id])+w_car_safe/2*math.sin(X_old[2,id]),
                 X_old[1,id]+l_car_safe_front/2*math.sin(X_old[2,id])-w_car_safe/2*math.cos(X_old[2,id])]])



        # plot the disturbance set from the perspective of the AV
        dist_comb = params.dist_comb
        w_ext = dist_comb[3]    # choosing [1 1]
        W_curr = w_ext * np.array([l_car / params.W_l_car_fac, w_car / params.W_w_car_fac])

            # count = 0
            # for car_id in range(0, params.num_cars):
            #     if id != car_id:
            #         for i in range(0, 2):
            #             X_old[i, car_id] = X_old[i, car_id] + W_curr[i] * Level_ratio[(id) * (params.num_cars - 1) + count][0]
            #         count += 1

        ego_car_id = 1 # AV id

        P0 = Level_ratio[ego_car_id * (params.num_cars-1) + count, 0]

        if params.sim_case == 0:
            W_curr *= 0
        elif params.sim_case == 1:
            W_curr *= P0


        dist_rect = np.array(
            [[X_old[0, id] - (l_car_safe+W_curr[0]) / 2 * math.cos(X_old[2, id]) + (w_car_safe+W_curr[1]) / 2 * math.sin(X_old[2, id]) - 0*W_curr[0],
              X_old[1, id] - (l_car_safe+W_curr[0]) / 2 * math.sin(X_old[2, id]) - (w_car_safe+W_curr[1]) / 2 * math.cos(X_old[2, id]) - 0*W_curr[1]],
             [X_old[0, id] - (l_car_safe+W_curr[0]) / 2 * math.cos(X_old[2, id]) - (w_car_safe+W_curr[1]) / 2 * math.sin(X_old[2, id]) - 0*W_curr[0],
              X_old[1, id] - (l_car_safe+W_curr[0]) / 2 * math.sin(X_old[2, id]) + (w_car_safe+W_curr[1]) / 2 * math.cos(X_old[2, id]) + 0*W_curr[1]],
             [X_old[0, id] + (l_car_safe+W_curr[0]) / 2 * math.cos(X_old[2, id]) - (w_car_safe+W_curr[1]) / 2 * math.sin(X_old[2, id]) + 0*W_curr[0],
              X_old[1, id] + (l_car_safe+W_curr[0]) / 2 * math.sin(X_old[2, id]) + (w_car_safe+W_curr[1]) / 2 * math.cos(X_old[2, id]) + 0*W_curr[1]],
             [X_old[0, id] + (l_car_safe+W_curr[0]) / 2 * math.cos(X_old[2, id]) + (w_car_safe+W_curr[1]) / 2 * math.sin(X_old[2, id]) + 0*W_curr[0],
              X_old[1, id] + (l_car_safe+W_curr[0]) / 2 * math.sin(X_old[2, id]) - (w_car_safe+W_curr[1]) / 2 * math.cos(X_old[2, id]) - 0*W_curr[1]]])

        # 
        if id == ego_car_id:
            color_id = 0
            # Disturbance rectangle
            
            plt.plot([rect[0, 0], rect[1, 0]], [rect[0, 1], rect[1, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([rect[1, 0], rect[2, 0]], [rect[1, 1], rect[2, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([rect[2, 0], rect[3, 0]], [rect[2, 1], rect[3, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([rect[0, 0], rect[3, 0]], [rect[0, 1], rect[3, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            count += 1
        else:
            plt.plot([dist_rect[0, 0], dist_rect[1, 0]], [dist_rect[0, 1], dist_rect[1, 1]], color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([dist_rect[1, 0], dist_rect[2, 0]], [dist_rect[1, 1], dist_rect[2, 1]], color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([dist_rect[2, 0], dist_rect[3, 0]], [dist_rect[2, 1], dist_rect[3, 1]], color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([dist_rect[0, 0], dist_rect[3, 0]], [dist_rect[0, 1], dist_rect[3, 1]], color=color[id], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([rect[0, 0], rect[1, 0]], [rect[0, 1], rect[1, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([rect[1, 0], rect[2, 0]], [rect[1, 1], rect[2, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([rect[2, 0], rect[3, 0]], [rect[2, 1], rect[3, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')
            plt.plot([rect[0, 0], rect[3, 0]], [rect[0, 1], rect[3, 1]], color=color[4], LineWidth=car_rect_lw+1,
                     linestyle='-.')

    #fig = plt.figure()
    # Setting axes limts
    x_lim = np.array([-2.5, 2.5])
    y_lim = np.array([-2.5, 2.5])
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(0)

    
    # # Display Car_id
    # for id in range(0, len(X_old[0,:])):
    #     ax.annotate(str(id+1), xy=(X_old[0, id], X_old[1, id]-0.5))

        
    #     #ax.annotate('v='+str(X_old[3,0])+'m/s', xy=(5, -10))    
    #     #ax.annotate('v='+str(X_old[3,1])+'m/s', xy=(5, 10))


        
    plt.yticks([])
    plt.xlabel('x (m)')
    # ax.axis('off')

    # plt.savefig(params.outdir+'/'+params.plot_fname+str(step)+plot_format, dpi=1200)
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()


