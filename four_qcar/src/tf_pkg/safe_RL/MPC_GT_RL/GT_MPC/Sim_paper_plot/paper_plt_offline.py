import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

color = ['r-', 'b-', 'y-', 'g-', 'm-', 'c-', 'k-.', 'k-']#['r--', 'b--', 'k--', 'c--', 'r-.', 'b-.', 'k-.', 'c-.']
labels_ls = ['Car# 1', 'Car# 2', 'Car# 3', 'Car# 4', 'Car# 5', 'Car# 6', 'Car# 7', 'Car# 8']
label_font_size = 20
time_range = []
def plot_vel(speed_data):
    data = np.array(speed_data)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(data[:, 0], data[:, 1], data[:, 0], data[:, 2], data[:, 0], data[:, 3])
    plt.show()   
    
def plot_level_ratio(Record_data, Level_ratio_his, path, params, Num_max):
    hunman_car =params.num_cars-1
    data = Record_data
    
    render =False
    if hunman_car == 3:
        size = (6, 6)
    elif hunman_car == 4:
        size = (7, 7)
    elif hunman_car == 5:
        size = (8, 8)
    elif hunman_car == 6:
        size = (9, 9)
    fig, ax = plt.subplots(hunman_car, 1, figsize=size)#, gridspec_kw={'width_ratios': [5, 3]}
    
    if render == True:
        data = np.array(data)
        for vehicle in range(hunman_car):
            ax[vehicle].plot(data[:, 0], Level_ratio_his[0:len(data[:, 0]),hunman_car + vehicle, 0], color[vehicle], label= labels_ls[vehicle], linewidth=3)
                # ax[0].set_xlabel("y (m)",fontsize=label_font_size)
            # ax[0].tick_params(labelsize=label_font_size)
            ax[vehicle].set_ylim([0, 1.1])
            ax[vehicle].grid(True)
            ax[vehicle].legend()
            ax[vehicle].tick_params(labelsize=label_font_size)
            # ax[1].legend()
        # ax[0].legend('Car# 1')
        # ax[1].legend('Car# 2')
        # ax[2].legend('Car# 3')
        ax[0].set_ylabel('$P^{(k=0)}[1]$',fontsize=label_font_size)
        ax[1].set_ylabel('$P^{(k=0)}[2]$',fontsize=label_font_size)
        ax[2].set_ylabel('$P^{(k=0)}[3]$',fontsize=label_font_size)
        if hunman_car >=4:
            ax[3].set_ylabel('$P^{(k=0)}[4]$',fontsize=label_font_size)
        if hunman_car >=5:
            ax[4].set_ylabel('$P^{(k=0)}[5]$',fontsize=label_font_size)
        if hunman_car >=6:
            ax[5].set_ylabel('$P^{(k=0)}[6]$',fontsize=label_font_size)
        if hunman_car >=7:
            ax[6].set_ylabel('$P^{(k=0)}[7]$',fontsize=label_font_size)
        ax[hunman_car-1].set_xlabel("time (sec)",fontsize=label_font_size)
        
        fig.tight_layout()
        
        # save_locat = '/home/sdcnlab025/ROS_test/four_qcar_Highway/src/tf_pkg/PaperTest/paper_results/'
        plt.savefig(path + '/' +'level_ratio'+ '.png', dpi = 100)#'.pdf', format="pdf", dpi=1200)
    else:
        _path = path +'ratios/'
        isExist = os.path.exists(_path)
        if not isExist:
            os.makedirs(_path)
        # x_ls = []
        # y_ls = []
        # for i in range(len(data)):
        #     x_ls.append(data[i][0])
        # print(x_ls)
        # for s in Level_ratio_his[0, 0:len(data)]:
        #     tem = []
        #     for j in range(hunman_car):
        #         tem.append(s[hunman_car + j, 0])
        #         # print(tem)
        #     y_ls.append(tem)
        # print(y_ls)
        data = np.array(data)
        for k in range(Num_max):
            for vehicle in range(hunman_car):
                if vehicle !=0:
                    _vehicle = vehicle+1
                else:
                    _vehicle = vehicle
                ax[vehicle].plot(data[0:k, 0], Level_ratio_his[0:k, hunman_car + vehicle, 0], color[_vehicle], label= labels_ls[vehicle], linewidth=3, linestyle = 'solid')  
                # ax[0].set_xlabel("y (m)",fontsize=label_font_size)
                
                # ax[0].tick_params(labelsize=label_font_size)
                ax[vehicle].set_ylim([0, 1.1])
                ax[vehicle].grid(True)
                ax[vehicle].legend()#fontsize = label_font_size
                ax[vehicle].tick_params(labelsize=label_font_size)
                ax[vehicle].yaxis.set_major_locator(MultipleLocator(0.5))
                # ax[vehicle].xaxis.set_major_locator(MultipleLocator(2))
                
                # ax[1].legend()
            # ax[0].legend('Car# 1')
            # ax[1].legend('Car# 2')
            # ax[2].legend('Car# 3')
            ax[0].set_ylabel('$P^{(k=0)}[1]$',fontsize=label_font_size)
            ax[1].set_ylabel('$P^{(k=0)}[2]$',fontsize=label_font_size)
            ax[2].set_ylabel('$P^{(k=0)}[3]$',fontsize=label_font_size)
            if hunman_car >=4:
                ax[3].set_ylabel('$P^{(k=0)}[4]$',fontsize=label_font_size)
            if hunman_car >=5:
                ax[4].set_ylabel('$P^{(k=0)}[5]$',fontsize=label_font_size)
            if hunman_car >=6:
                ax[5].set_ylabel('$P^{(k=0)}[6]$',fontsize=label_font_size)
            if hunman_car >=7:
                ax[6].set_ylabel('$P^{(k=0)}[7]$',fontsize=label_font_size)
            ax[hunman_car-1].set_xlabel("time (sec)",fontsize=label_font_size)
           
                
            
            fig.tight_layout()
            # save_locat = '/home/sdcnlab025/ROS_test/four_qcar_Highway/src/tf_pkg/PaperTest/paper_results/'
            plt.savefig(_path + '/' +str(k)+'.png', dpi = 100)#'.pdf', format="pdf", dpi=1200)
            for i in range(hunman_car):
                ax[i].cla()
        
            
        # plt.show()   

def plot_runtime(Record_data1, Record_data2, path_ls):
    render = True
    if render==False:
        fig, ax = plt.subplots(1, 1, figsize=(6,4))#, gridspec_kw={'width_ratios': [5, 3]}
        i = 0
        data1 = Record_data1
        data2 = Record_data2


        plt.plot(data1[:, 0], data1[:,1], '-.', label = 'Our Scalable Level-k', linewidth=3)

        plt.plot(data2[:, 0], data2[:,1], '-', label = 'Traditional Level-k', linewidth=3)
        time_range.append(data1[-1, 0])
        try:
            plt.xlim([0, min(time_range)])
            plt.ylim([0, 0.45])
            plt.xlabel('time (sec)',fontsize=label_font_size)
            plt.ylabel('Computational time (sec)', fontsize=label_font_size)
        except:
            pass
        plt.legend()
        ax.grid(True)                # plt.savefig(save_locat + str(test_case) + '.pdf', format="pdf", dpi=1200)
        ax.yaxis.set_major_locator(MultipleLocator(0.15))
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.tick_params(labelsize=label_font_size)
        fig.tight_layout()
        plt.savefig(path_ls+'/'+'runing_time'+'.png', dpi = 100)#, bbox_inches='tight', pad_inches=0.5
                        # plt.show()
    else:
        _path = path_ls +'/' +'run_time/'
        isExist = os.path.exists(_path)
        if not isExist:
            os.makedirs(_path)
        data1 = Record_data1
        data2 = Record_data2
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        
        for i in range(len(Record_data1)):
            plt.plot(data1[0:i, 0], data1[0:i, 1], ':', label = 'Our Scalable Level-k', linewidth=3)
            plt.plot(data2[:, 0], data2[:, 1], ':', label = 'Traditional Level-k', linewidth=3)
            x_lim = data1[i, 0]
            plt.xlim([0, x_lim])
            plt.ylim([0, 0.45])
            plt.xlabel('time (sec)',fontsize=label_font_size)
            plt.ylabel('Computational time (sec)', fontsize=label_font_size)
            plt.legend()
            ax.grid(True)                # plt.savefig(save_locat + str(test_case) + '.pdf', format="pdf", dpi=1200)
            ax.yaxis.set_major_locator(MultipleLocator(0.15))
            ax.xaxis.set_major_locator(MultipleLocator(4))
            ax.tick_params(labelsize=label_font_size)
            fig.tight_layout()
            plt.savefig(_path + '/' +str(i)+'.png', dpi = 100)
            #for i in range(hunman_car):
            ax.cla()
    
def plot_progress(Record_data, _vel, path_ls, params):
    hunman_car = params.num_cars-1
    fig, ax = plt.subplots(1, 1, figsize=(6,4))#, gridspec_kw={'width_ratios': [5, 3]}
    _vel = np.array(_vel)
    data = Record_data
    data = np.array(data)
    short = min([len(data), len(_vel)])
    plt.plot(data[0:short, 0], _vel[0:short,0], 'r--', label = 'Car# 1', linewidth=3)
    plt.plot(data[0:short, 0], _vel[0:short,1], 'b--', label = 'Ego', linewidth=3)
    plt.plot(data[0:short, 0], _vel[0:short,2], 'y--', label = 'Car# 2', linewidth=3)
    plt.plot(data[0:short, 0], _vel[0:short,3], 'g--', label = 'Car# 3', linewidth=3)
    
    if hunman_car >= 4:
        plt.plot(data[:, 0], _vel[:,4], 'm--', label = 'Car# 4', linewidth=3)
    if hunman_car >= 5:
        plt.plot(data[:, 0], _vel[:,5], 'c--', label = 'Car# 5', linewidth=3)
    if hunman_car >= 6:
        plt.plot(data[:, 0], _vel[:,6], 'k--', label = 'Car# 6', linewidth=3)
    if hunman_car >= 7:
        plt.plot(data[:, 0], _vel[:,7], 'm--', label = 'Car# 7', linewidth=3)

    ax.grid(True)                # plt.savefig(save_locat + str(test_case) + '.pdf', format="pdf", dpi=1200)
    plt.xlim([0.0, data[-1, 0]])
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_locator(MultipleLocator(4.0))
    
    ax.tick_params(labelsize=label_font_size)
    ax.set_xlabel('time (sec)',fontsize=label_font_size)
    ax.set_ylabel('passed waypoints',fontsize=label_font_size)
    plt.legend()
    ax.tick_params(labelsize=label_font_size)
    fig.tight_layout()

    plt.savefig(path_ls+'progress'+'.png', dpi = 100)#, bbox_inches='tight', pad_inches=0.5
    # plt.close()
