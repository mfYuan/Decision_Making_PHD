import cv2
import os


def save_plot(params, step):   
    #outfile = 'Test'+'.mp4',
    plot_fname = 'plot',
    plot_format = '.png',
    outdir = 'Images',
                #/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded1/Scalable_result############### Graphs  motions  ratios  run_time
    out_fname = '/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded1/Scalable_result/Videos/111011201020020000001_7cars_video_run6.mp4'#'/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded/61112220102012/motion/1.mp4' 

    plot_fname = '/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded1/Scalable_result/111011201020020000001_7cars_video/txtfile/figs/run_time/'#'/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded/61112220102012/motion/'
    imgs_ls = os.listdir(plot_fname)#

    img_array = []
    
    if os.path.exists(out_fname):
        os.remove(out_fname)
    
    for i in range(len(imgs_ls)):
        filename = plot_fname+str(i)+'.png'
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
 
    out = cv2.VideoWriter(out_fname,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 6, size) #14
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

save_plot(params=1, step=1)