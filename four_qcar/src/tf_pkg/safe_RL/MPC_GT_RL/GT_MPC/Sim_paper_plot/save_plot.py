import cv2
import os


def save_plot(params, step):   
    #outfile = 'Test'+'.mp4',
    plot_fname = 'plot',
    plot_format = '.png',
    outdir = 'Images',
                
    out_fname = ['E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/Videos/333221220002_motion.mp4', 
    'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/Videos/333221220002_graph.mp4', 
    'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/Videos/333221220002_level.mp4' ]

    # plot_fname = 'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/333221220002/txtfile/figs/motions/'
    plot_fname = ['E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/333221220002/txtfile/figs/motions/', 
    'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/333221220002/txtfile/figs/Graphs/', 
    'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/333221220002/txtfile/figs/ratios/']
    




    for item in range(len(plot_fname)):

        imgs_ls = os.listdir(plot_fname[item])

        img_array = []
        
        if os.path.exists(out_fname[item]):
            os.remove(out_fname[item])
        
        for i in range(len(imgs_ls)):
            filename = plot_fname[item]+str(i)+'.png'
            print(filename)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
    
    
        out = cv2.VideoWriter(out_fname[item],cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 11, size)
    
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

save_plot(params=1, step=1)