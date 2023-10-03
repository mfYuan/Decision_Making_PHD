import numpy as np
import scipy.linalg
import itertools
from scipy.spatial import cKDTree
import csv

def get_params():
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # Declare constant parameters
    params = Bunch(
                prediction_horizon = 2, # 10
                w_lane = 0.4,          # (m) lane width
                l_car = 0.425,          # (m) car length
                w_car = 0.192,          # (m) car width
                l_road = 4000,       # (m)
                v_nominal = 15.0/36.0,  # (m/s) nominal car speed  
                v_max = 20.0/36.0,     # (m/s) maximum car speed 
                v_min = 0.0/36.0,      # (m/s) minimum car speed
                t_step_DT = 0.65,     # (s) 
                t_step_DT_2 = 0.7,#0.65,   # (s)  # move blocked
                t_step_Sim = 0.7,#0.65,    # (s)             #0.25
                discount = 0.8,      # discount factor # 0.8
                dR_drop = -1e9,      # ?
                num_cars = 4,        # number of cars
                num_AV = 1,
                num_Human =-1,
                max_episode = 1,     # number of maximum episode
                num_lanes =3,        # number of lanes
                init_x_range = 30,
                episode = 0,
                lr = 0.2125,
                lf = 0.2125,
                v_target = 15.0/36.0,
                outfile = 'Test.mp4',
                plot_fname = 'plot',
                plot_format = '.jpg',
                outdir = 'Images',
                max_acc = 0.25/1,
                max_dec = 0.3/1,
                maintain_acc = 0.0,
                fps = 3,
                sim_case = 1,
                l_car_safe_fac = 1.6,#1.6, # 1.1 ratio of length
                w_car_safe_fac = 1.0, # 1.25 ratio of width
                l_car_safe_fac_tp = 2.0, # 1.1 ratio of length
                w_car_safe_fac_tp = 1.5, # 1.25 ratio of width
                W_l_car_fac = 1.6,#1.6, # 1.5
                W_w_car_fac = 0, # 3
                render = False
                )

    params.complete_flag = np.zeros((params.max_episode,params.num_cars))
    folder_path = '/home/sdcnlab025/ROS_test/four_qcar/src/waypoint_loader/waypoints/'
    path = []
    
    for i in range(1, 13):
        file_name = folder_path + 'waypoints'+str(i) + '.csv'       
        pos_list = []
        
        with open(file_name, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                pos_list.append(list(map(float, row[0:2])))
            pos_list = np.array(pos_list)
        path.append(pos_list)
    path = np.array(path)
    waypoints = {}
    KDtrees_12 = {}
    for i in range(1, 13):
        KDtrees_12[str(i)] = cKDTree(path[i-1])
        waypoints[str(i)] = path[i-1]
        
    # Disturbances only in inputs
    # Combinations of disturbances = 2^(m)
    params.dist_comb = [(1, 1)] #list(itertools.product([-1, 1], repeat=2)) # [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    params.KDtrees_12 = KDtrees_12
    params.waypoints = waypoints
        
    folder_path_shift = '/home/sdcnlab025/ROS_test/four_qcar/src/waypoint_loader/waypoints/shift/'
    _ls = ['r', 'rr', 'l', 'll']
    waypoints_all ={}                                                                                                                    
    for ele in _ls:
        path = []
        for i in range(1, 13):
            file_name = folder_path_shift + 'waypoints'+str(i)+ ele + '.csv'       
            pos_list = []
            
            with open(file_name, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    pos_list.append(list(map(float, row[0:2])))
                pos_list = np.array(pos_list)
            path.append(pos_list)
        path = np.array(path)
        waypoints = {}
        for i in range(1, 13):
            waypoints[str(i)] = path[i-1]
        waypoints_all[ele] = waypoints

    params.waypoints_shift = waypoints_all
    
    # Intersection background
    folder_path_background = '/home/sdcnlab025/ROS_test/four_qcar/src/waypoint_loader/waypoints/Background/'
    _path = []
    for i in range(1, 11):
        _file_name = folder_path_background +str(i)+ '.csv'  
        # print(file_name)         
        pos_list = []
        with open(_file_name, 'r') as _csvfile:
            _spamreader = csv.reader(_csvfile, delimiter=',')
            for row in _spamreader:
                pos_list.append(list(map(float, row[0:2])))
            pos_list = np.array(pos_list)
        _path.append(pos_list)
    _path = np.array(_path)
    intersection = {}
    for i in range(1, 11):
        intersection[str(i)] = _path[i-1]

    params.inter = intersection
    return params
