import matplotlib.pyplot as plt
import numpy as np

color = ['r--', 'b--', 'k--', 'c--', 'r-.', 'b-.', 'k-.', 'c-.']
labels_ls = ['Car# 1', 'Car# 2', 'Car# 3', 'Car# 4', 'Car# 5', 'Car# 6', 'Car# 7', 'Car# 8']
label_font_size = 15
time_range = []
def plot_vel(speed_data):
    data = np.array(speed_data)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(data[:, 0], data[:, 1], data[:, 0], data[:, 2], data[:, 0], data[:, 3])
    plt.show()   
    
def plot_level_ratio(Record_data, Level_ratio_his, path, ls, params):
    hunman_car =params.num_cars-1
    data = Record_data[str(ls[-1])]
    
    render =False
    fig, ax = plt.subplots(hunman_car, 1, figsize=(6,6))#, gridspec_kw={'width_ratios': [5, 3]}
    
    if render == True:
        data = np.array(data)
        for vehicle in range(hunman_car):
            ax[vehicle].plot(data[:, 0], Level_ratio_his[0, 0:len(data[:, 0]),hunman_car + vehicle, 0], color[vehicle], label= labels_ls[vehicle])
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
        if hunman_car ==4:
            ax[3].set_ylabel('$P^{(k=0)}[4]$',fontsize=label_font_size)
            ax[3].set_xlabel("time (sec)",fontsize=label_font_size)
        else:
            ax[2].set_xlabel("time (sec)",fontsize=label_font_size)
            
        
        fig.tight_layout()
        
        # save_locat = '/home/sdcnlab025/ROS_test/four_qcar_Highway/src/tf_pkg/PaperTest/paper_results/'
        plt.savefig(path[2] + '/' +'level_ratio'+ '.png', dpi = 100)#'.pdf', format="pdf", dpi=1200)
    else:
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
        for k in range(len(data)):
            for vehicle in range(hunman_car):
                ax[vehicle].plot(data[0:k, 0], Level_ratio_his[0, 0:k, hunman_car + vehicle, 0], label= labels_ls[vehicle])
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
            if hunman_car ==4:
                ax[3].set_ylabel('$P^{(k=0)}[4]$',fontsize=label_font_size)
            elif hunman_car==5:
                ax[3].set_ylabel('$P^{(k=0)}[4]$',fontsize=label_font_size)
                ax[4].set_ylabel('$P^{(k=0)}[5]$',fontsize=label_font_size)
            elif hunman_car==6:
                ax[3].set_ylabel('$P^{(k=0)}[4]$',fontsize=label_font_size)
                ax[4].set_ylabel('$P^{(k=0)}[5]$',fontsize=label_font_size)
                ax[5].set_ylabel('$P^{(k=0)}[6]$',fontsize=label_font_size)
            elif hunman_car==7:
                ax[3].set_ylabel('$P^{(k=0)}[4]$',fontsize=label_font_size)
                ax[4].set_ylabel('$P^{(k=0)}[5]$',fontsize=label_font_size)
                ax[5].set_ylabel('$P^{(k=0)}[6]$',fontsize=label_font_size)
                ax[6].set_ylabel('$P^{(k=0)}[7]$',fontsize=label_font_size)
               
          
            ax[hunman_car-1].set_xlabel("time (sec)",fontsize=label_font_size)
                
            
            
            # save_locat = '/home/sdcnlab025/ROS_test/four_qcar_Highway/src/tf_pkg/PaperTest/paper_results/'
            plt.savefig(path[3] + '/' +str(k)+'.png', dpi = 100)#'.pdf', format="pdf", dpi=1200)
            for i in range(hunman_car):
                ax[i].cla()
            fig.tight_layout()
            
        # plt.show()   

def plot_runtime(Record_data, ls, path_ls, params):

    fig, ax = plt.subplots(1, 1, figsize=(6,6))#, gridspec_kw={'width_ratios': [5, 3]}
    i = 0
    for item in ls:
        data = Record_data[str(item)]
        data = np.array(data)
        if i == 0:
            plt.plot(data[:, 0], data[:,1], '-.', label = 'Traditional Level-k')
            i +=1
        else:
            plt.plot(data[:, 0], data[:,1], '-.', label = 'Our Scalable Level-k')
        time_range.append(data[-1, 0])
        try:
            plt.xlim([0, min(time_range)])
            plt.ylim([0, 0.35])
            plt.xlabel('time (sec)',fontsize=label_font_size)
            plt.ylabel('Computational time (sec)', fontsize=label_font_size)
        except:
            pass
    plt.legend()
    ax.grid(True)                # plt.savefig(save_locat + str(test_case) + '.pdf', format="pdf", dpi=1200)
    ax.tick_params(labelsize=label_font_size)
    fig.tight_layout()
    plt.savefig(path_ls+'/'+'runing_time'+'.png', dpi = 100)#, bbox_inches='tight', pad_inches=0.5
                    # plt.show()
    
def plot_progress(Record_data, _vel, path_ls, ls, params):
    hunman_car = params.num_cars-1
    fig, ax = plt.subplots(1, 1, figsize=(6,6))#, gridspec_kw={'width_ratios': [5, 3]}
    _vel = np.array(_vel)
    data = Record_data[str(ls[-1])]
    data = np.array(data)
    plt.plot(data[:, 0], _vel[:,0], 'r--', label = 'Car# 1')
    plt.plot(data[:, 0], _vel[:,1], 'k--', label = 'Ego')
    plt.plot(data[:, 0], _vel[:,2], 'b--', label = 'Car# 2')
    plt.plot(data[:, 0], _vel[:,3], 'c--', label = 'Car# 3')
    if hunman_car == 4:
        plt.plot(data[:, 0], _vel[:,4], 'm--', label = 'Car# 4')
    ax.grid(True)                # plt.savefig(save_locat + str(test_case) + '.pdf', format="pdf", dpi=1200)
    ax.tick_params(labelsize=label_font_size)
    ax.set_xlabel('time (sec)',fontsize=label_font_size)
    ax.set_ylabel('passed waypoints',fontsize=label_font_size)
    plt.legend()
    ax.tick_params(labelsize=label_font_size)
    fig.tight_layout()
    plt.savefig(path_ls[2]+'/'+'progress'+'.png', dpi = 100)#, bbox_inches='tight', pad_inches=0.5
    # plt.close()
