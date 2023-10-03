import numpy as np

def initial():
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    traffic = Bunch( 
        x = np.empty(shape=[1, 0]),     
        y = np.empty(shape=[1, 0]), 
        orientation = np.empty(shape=[1, 0]),
        v_car = np.empty(shape=[1, 0]), 
        AV_flag = np.empty(shape=[1, 0]),
        Final_x = np.empty(shape=[1, 0]),
        Final_y = np.empty(shape=[1, 0]),
        path_id = np.empty(shape=[1, 0]),
        Currend_index = np.empty(shape=[1, 0]),
        Driver_level = np.empty(shape=[1, 0])
        )
    return traffic

def update(traffic, x_car, y_car, orientation_car, v_car, AV_flag, Final_x, Final_y, path_id, Currend_index, Driver_level):
    traffic.x = np.append(traffic.x, [[x_car]], axis=1 )
    traffic.y = np.append(traffic.y, [[y_car]], axis=1)
    traffic.orientation = np.append(traffic.orientation, [[orientation_car]], axis=1)
    traffic.v_car = np.append(traffic.v_car, [[v_car]], axis=1)
    traffic.AV_flag = np.append(traffic.AV_flag, [[AV_flag]], axis=1)
    traffic.Final_x = np.append(traffic.Final_x, [[Final_x]], axis=1)
    traffic.Final_y = np.append(traffic.Final_y, [[Final_y]], axis=1)
    traffic.path_id = np.append(traffic.path_id, [[path_id]], axis=1)
    traffic.Currend_index = np.append(traffic.Currend_index, [[Currend_index]], axis=1)
    traffic.Driver_level = np.append(traffic.Driver_level, [[Driver_level]], axis=1)
    return traffic
