import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from scipy import ndimage
import os

from PIL import Image
from PIL import ImageChops
debug = False
label_font_size = 20
#test_id = 4
def plot_graph(X_old, matrix, fig, counter, path):
    _path = path + 'Graphs'
    isExist = os.path.exists(_path)
    if not isExist:
        os.makedirs(_path)
    #plt.figure(fig.number)
    # plt.cla()
    #ax = plt.gca()
    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    color = ['r', 'b','y', 'g', 'm','c', 'k']
    #color = ['k']*len(X_old[0][:])
    name_ls = []
     
    for test_id in range(len(X_old[0][:])):
        if matrix[test_id, test_id]>0:
            for i in range(test_id+1, len(matrix[test_id])):
                if matrix[test_id][i] ==-1:
                    if matrix[i][test_id] ==-1:                      
                        plt.annotate('', xy=(-X_old[1][i], X_old[0][i]),xytext=(-X_old[1][test_id], X_old[0][test_id]),arrowprops=dict(facecolor='black', lw=3, arrowstyle='<->'), size=label_font_size)
                     
    # if matrix[test_id, test_id]>0:
    #     for i in range(len(matrix[test_id])):
    #         if matrix[test_id][i] ==-1:
    #             if i ==1:
    #                 name = 'Ego'
    #             elif i < 1:
    #                 name = 'Car 1'
    #             else:
    #                 name = 'Car '+ str(i)
                
    #             plt.annotate(name, xy=(X_old[0][i],X_old[1][i]),xytext=(X_old[0][test_id],X_old[1][test_id]),arrowprops={'arrowstyle':'->'})

    
    # for id in range(0, len(X_old[0,:])):
    #     if matrix[id, id]>0:
    #         for i in range(len(matrix[id])):
    #             if matrix[id][i] ==-1:
    #                 if i ==1:
    #                     name = 'Ego'
    #                 elif i < 1:
    #                     name = 'Car 1'
    #                 else:
    #                     name = 'Car '+ str(i)
                    
    #                 plt.annotate(name, xy=(X_old[0][i],X_old[1][i]),xytext=(X_old[0][id],X_old[1][id]),arrowprops={'arrowstyle':'->'})

    for test_id in range(len(X_old[0][:])):
        # Display Car_id and circle
        # print(test_id)
        name_ls.append('car'+str(test_id))
        name_ls[test_id] = plt.Circle((-X_old[1][test_id], X_old[0][test_id],), 0.18, color=color[test_id], fill=False, linewidth=3) 
        if test_id ==0:
            ax.annotate('Car 1', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        elif test_id ==1:
            ax.annotate('Ego', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        elif test_id ==2:
            ax.annotate('Car 2', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        elif test_id ==3:
            ax.annotate('Car 3', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        elif test_id ==4:
            ax.annotate('Car 4', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        elif test_id ==5:
            ax.annotate('Car 5', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        elif test_id ==6:
            ax.annotate('Car 6', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        elif test_id ==7:
            ax.annotate('Car 7', xy=(-X_old[1, test_id], X_old[0, test_id]-0.3), size=label_font_size)
        
    for car in name_ls:
        ax.add_patch(car)

    #fig = plt.figure()
    # Setting axes limts
    
    

    #
    ax.set_xticks([])
    ax.set_yticks([])

    #
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    limit_size = 2.0
    x_lim = np.array([-limit_size, limit_size])
    y_lim = np.array([-limit_size, limit_size])
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    plt.tight_layout()

    # # Display Car_id
    # for id in range(0, len(X_old[0,:])):
    #     ax.annotate(str(id+1), xy=(X_old[0, id], X_old[1, id]-0.5))

        
    #     #ax.annotate('v='+str(X_old[3,0])+'m/s', xy=(5, -10))    
    #     #ax.annotate('v='+str(X_old[3,1])+'m/s', xy=(5, 10))


        
    # plt.yticks([])
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    #ax.tick_params(labelsize=label_font_size)
    # ax.axis('off')
    
    plt.savefig(_path+'/'+str(counter)+'.png', dpi = 100)#, bbox_inches='tight', pad_inches=0.5)
    # plt.savefig('/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Graph/'+str(counter)+'.png', dpi = 100, bbox_inches='tight', pad_inches=0.5)
    #plt.show(block=False)
    ax.cla()


