import environment
debug = 1
# robust decision tree

# level 0 driver assumes that all other cars are static
def initial_Q_value(action_space, prediction_horizon, Q_init):
     # initial Q value
    Q_value = [[Q_init]] * action_space # Q value for each action
    if prediction_horizon == 2:
        Q_value_2_min = [[Q_init] * action_space for i in range(action_space)]
    elif prediction_horizon == 3:
        Q_value_2_min = [[[Q_init] * action_space for i in range(action_space)] for i in range(action_space)]
    elif prediction_horizon == 4:
        Q_value_2_min = [[[[Q_init] * action_space for i in range(action_space)] for i in range(action_space)] for i in range(action_space)]
    elif prediction_horizon == 5:
        Q_value_2_min = [[[[[Q_init] * action_space for i in range(action_space)] for i in range(action_space)] for i in range(action_space)] for i in range(action_space)]
    return Q_value, Q_value_2_min
# Decision Tree action search
def decisiontree_l0(X_old, car_id, action_space, params, Level_ratio):
    prediction_horizon = params.prediction_horizon
    if debug == 1:
        Q_init = -1e6
        X_old1 = X_old.copy() # copy the old state
        discount = params.discount  # discount factor
        dR_drop = params.dR_drop #
        t_step_DT = params.t_step_DT #prediction time step
        
        action_id = [[]] * action_space.size 
        Buffer = [[]] * 10#prediction_horizon
        Q_value, Q_value_2_min = initial_Q_value(action_space.size, prediction_horizon, Q_init) # Q value for each action

        R1_max, R2_max, R3_max, R4_max, R5_max = -1e10, -1e10, -1e10, -1e10, -1e10
        dist_comb = [(1, 1)]#params.dist_comb
        Buffer[0] = X_old1
        dist_id_1, dist_id_2 = dist_comb[0][0], dist_comb[0][1]
        if X_old1[4, car_id] == 1: # if the car is AV
            for id_1 in range(0, action_space.size): # for each action
                k = 0
                X_old1 = Buffer[k] # get the old state
                # get the new state and reward
                
                X_new, R1 = environment.environment(X_old1, car_id, id_1, t_step_DT, params, dist_id_1, Level_ratio)
                # add the new state to the buffer
                Buffer[k + 1] = X_new
        
                Q_value_2 = [[Q_init]  for i in range (action_space.size)]

                for id_2 in range(0, action_space.size):
                    
                    k = 1
                    X_old1 = Buffer[k]
                    X_new, R2 = environment.environment(X_old1, car_id, id_2, t_step_DT, params, dist_id_2, Level_ratio)

                    if Q_value_2[id_2][0] == Q_init:
                        Q_value_2[id_2] = list([R2 * discount**(prediction_horizon-1)])
                    else:
                        #Q_value_2[id_2] = Q_value_2[id_2] + list([R2 * discount])
                        print('it is not supposed to be having value here')
                    Buffer[k + 1] = X_new
                    Q_value_3 = [[Q_init]  for i in range (action_space.size)]
                    if prediction_horizon >= 3:
                        for id_3 in range(0, action_space.size):
                            k = 2
                            X_old1 = Buffer[k]
                            X_new, R3 = environment.environment(X_old1, car_id, id_3, t_step_DT, params, dist_id_2, Level_ratio)
                            if Q_value_3[id_3][0] == Q_init:
                                Q_value_3[id_3] = list([R3 * discount**(prediction_horizon-1)])
                            else:   
                                Q_value_3[id_3] = Q_value_3[id_3] + list([R3 * discount])
                                print('it is not supposed to be having value here')
                            Buffer[k + 1] = X_new
                            Q_value_4 = [[Q_init]  for i in range (action_space.size)]
                            
                            if prediction_horizon >= 4:
                                for id_4 in range(0, action_space.size):
                                    k = 3
                                    X_old1 = Buffer[k]
                                    X_new, R4 = environment.environment(X_old1, car_id, id_4, t_step_DT, params, dist_id_2, Level_ratio)
                                    if Q_value_4[id_4][0] == Q_init:
                                        Q_value_4[id_4] = list([R4 * discount**(prediction_horizon-1)])
                                    else:
                                        Q_value_4[id_4] = Q_value_4[id_4] + list([R4 * discount])
                                        print('it is not supposed to be having value here')
                                    #Buffer[k + 1] = X_new
                                    #Q_value_5 = [[Q_init]  for i in range (action_space.size)]
                                Q_value_2_min[id_1][id_2][id_3][id_4] = [Q_value_4[id_4][0]+Q_value_3[id_3][0]+Q_value_2[id_2][0]+R1]
                                
                        Q_value_2_min[id_1][id_2][id_3] = [Q_value_3[id_3][0]+Q_value_2[id_2][0]+R1]
                    
                    Q_value_2_min[id_1][id_2] = Q_value_2[id_2]

                Q_value[id_1] = Q_value_2_min[id_1]
            #print(Q_value)
            Q_value_opt = [[]] * action_space.size
            index_opt = [[]] * action_space.size
            for id in range(0, action_space.size):
                Q_value_opt[id] = max(Q_value[id])
                index_opt[id] = Q_value[id].index(max(Q_value[id]))
            
            id_opt = Q_value_opt.index(max(Q_value_opt))

            Action_id = list([id_opt, index_opt[id_opt]])
            print(Action_id)
            print('----------------')
        else:
            dist_id = 1  # dummy value
            for id_1 in range(0, action_space.size):
                k = 0
                X_old1 = Buffer[k]
                X_new, R1 = environment.environment(X_old1, car_id, id_1, t_step_DT, params, dist_id, Level_ratio)
                R1_max = max(R1_max, R1)
                if R1 < R1_max + dR_drop:
                    continue
                Buffer[k + 1] = X_new

                for id_2 in range(0, action_space.size):
                    k = 1
                    X_old1 = Buffer[k]
                    X_new, R2 = environment.environment(X_old1, car_id, id_2, t_step_DT, params, dist_id, Level_ratio)
                    R2_max = max(R2_max, R2)
                    if R2 < R2_max + dR_drop:
                        continue

                    if Q_value[id_1][0] == Q_init:
                        Q_value[id_1] = [R1 + R2 * discount]
                    else:
                        Q_value[id_1] = Q_value[id_1] + list([R1 + R2 * discount])
                        
                    if action_id[id_1] == []:
                        action_id[id_1] = [[id_1, id_2]]
                    else:
                        action_id[id_1] = action_id[id_1] + list([[id_1, id_2]])

            Q_value_opt = [[]] * action_space.size
            index_opt = [[]] * action_space.size
            for id in range(0, action_space.size):
                Q_value_opt[id] = max(Q_value[id])
                index_opt[id] = Q_value[id].index(max(Q_value[id]))

            id_opt = Q_value_opt.index(max(Q_value_opt))

            Action_id = action_id[id_opt][index_opt[id_opt]]
            
        return Q_value_opt, Action_id
    else:
        X_old1 = X_old.copy() # copy the old state
        discount = params.discount  # discount factor
        dR_drop = params.dR_drop #
        t_step_DT = params.t_step_DT #prediction time step
        
        Q_init = -1e6 # initial Q value
        Q_value = [[Q_init]] * action_space.size # Q value for each action
        action_id = [[]] * action_space.size 
        Buffer = [[]] * 3
        R1_max, R2_max, R3_max, R4_max, R5_max = -1e10, -1e10, -1e10, -1e10, -1e10
        dist_comb = [(1, 1)]#params.dist_comb


        Buffer[0] = X_old1
        Q_value_2_min = [[[[Q_init]] * action_space.size for i in range(len(dist_comb))] for i in range(action_space.size)] 

        if X_old1[4, car_id] == 1: # if the car is AV
            for id_1 in range(0, action_space.size): # for each action
                for dist_id_1 in range(0, len(dist_comb)): # for each disturbance combination
                    k = 0
                    X_old1 = Buffer[k] # get the old state
                    # get the new state and reward
                    
                    X_new, R1 = environment.environment(X_old1, car_id, id_1, t_step_DT, params, dist_id_1, Level_ratio)
                    # add the new state to the buffer
                    Buffer[k + 1] = X_new
                    Q_value_2 = [[Q_init]  for i in range (action_space.size)]

                    for id_2 in range(0, action_space.size):
                        for dist_id_2 in range(0, len(dist_comb)):
                            k = 1
                            X_old1 = Buffer[k]
                            X_new, R2 = environment.environment(X_old1, car_id, id_2, t_step_DT, params, dist_id_2, Level_ratio)

                            if Q_value_2[id_2][0] == Q_init:
                                Q_value_2[id_2] = list([R2 * discount])
                            else:#useless if there is only one disturbance
                                Q_value_2[id_2] = Q_value_2[id_2] + list([R2 * discount])    
                                print('Double-check this part if there is only one disturbance')
                        
                        Q_value_2_min[id_1][dist_id_1][id_2] = [min(Q_value_2[id_2][:]) + R1]
                
                Q_value[id_1] = min(Q_value_2_min[id_1][:])                       


            Q_value_opt = [[]] * action_space.size
            index_opt = [[]] * action_space.size
            for id in range(0, action_space.size):
                Q_value_opt[id] = max(Q_value[id])
                index_opt[id] = Q_value[id].index(max(Q_value[id]))
            print(Q_value)

            id_opt = Q_value_opt.index(max(Q_value_opt))

            Action_id = list([id_opt, index_opt[id_opt]])
            print(Action_id)
        else:
            dist_id = 1  # dummy value
            for id_1 in range(0, action_space.size):
                k = 0
                X_old1 = Buffer[k]
                X_new, R1 = environment.environment(X_old1, car_id, id_1, t_step_DT, params, dist_id, Level_ratio)
                R1_max = max(R1_max, R1)
                if R1 < R1_max + dR_drop:
                    print("dR_drop1")
                    continue
                Buffer[k + 1] = X_new

                for id_2 in range(0, action_space.size):
                    k = 1
                    X_old1 = Buffer[k]
                    X_new, R2 = environment.environment(X_old1, car_id, id_2, t_step_DT, params, dist_id, Level_ratio)
                    R2_max = max(R2_max, R2)
                    if R2 < R2_max + dR_drop:
                        continue

                    if Q_value[id_1][0] == Q_init:
                        Q_value[id_1] = [R1 + R2 * discount]
                    else:
                        Q_value[id_1] = Q_value[id_1] + list([R1 + R2 * discount])
                        
                    if action_id[id_1] == []:
                        action_id[id_1] = [[id_1, id_2]]
                    else:
                        action_id[id_1] = action_id[id_1] + list([[id_1, id_2]])

            Q_value_opt = [[]] * action_space.size
            index_opt = [[]] * action_space.size
            for id in range(0, action_space.size):
                Q_value_opt[id] = max(Q_value[id])
                index_opt[id] = Q_value[id].index(max(Q_value[id]))

            id_opt = Q_value_opt.index(max(Q_value_opt))

            Action_id = action_id[id_opt][index_opt[id_opt]]
            
        return Q_value_opt, Action_id
        