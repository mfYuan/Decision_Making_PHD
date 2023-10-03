import environment


# Decision Tree action search
def DecisionTree_L1(X_pseudo, car_id, action_space, params, Level_ratio):
    X_pseudo1 = X_pseudo.copy()
    discount = params.discount  # discount factor
    dR_drop = params.dR_drop
    t_step_DT = params.t_step_DT
    Q_init = -1e6
    Q_value = [[Q_init]] * action_space.size
    action_id = [[]] * action_space.size
    Buffer = [[]] * 3
    R1_max, R2_max, R3_max = -1e10, -1e10, -1e10
    R1_min, R2_min, R3_min = 1e10, 1e10, 1e10
    dist_comb = params.dist_comb

    Q_value_2_min = [[[[Q_init]] * action_space.size for i in range(len(dist_comb))] for i in range(action_space.size)]

    Buffer[0] = X_pseudo1[0]

    if X_pseudo1[0, 4, car_id] == 1:
        for id_1 in range(0, action_space.size):

            for dist_id_1 in range(0, len(dist_comb)):
                k = 0
                X_old1 = Buffer[k]
                X_new, R1 = environment.environment(X_old1, car_id, id_1, t_step_DT, params, dist_id_1, Level_ratio)
                # if R1 < R1_max +dR_drop:
                #     continue

                Buffer[k + 1] = X_pseudo1[k + 1]
                Buffer[k + 1][:, car_id] = X_new[:, car_id]

                Q_value_2 = [[Q_init]  for i in range (action_space.size)]

                for id_2 in range(0, action_space.size):

                    for dist_id_2 in range(0, len(dist_comb)):
                        k = 1
                        X_old1 = Buffer[k]
                        X_new, R2 = environment.environment(X_old1, car_id, id_2, t_step_DT, params, dist_id_2,
                                                            Level_ratio)

                        if Q_value_2[id_2][0] == Q_init:
                            Q_value_2[id_2] = list([R2 * discount])
                        else:
                            Q_value_2[id_2] = Q_value_2[id_2] + list([R2 * discount])


                    Q_value_2_min[id_1][dist_id_1][id_2] = [min(Q_value_2[id_2][:]) + R1]


            Q_value[id_1] = min(Q_value_2_min[id_1][:])

        Q_value_opt = [[]] * action_space.size
        index_opt = [[]] * action_space.size
        for id in range(0, action_space.size):
            Q_value_opt[id] = max(Q_value[id])
            index_opt[id] = Q_value[id].index(max(Q_value[id]))

        id_opt = Q_value_opt.index(max(Q_value_opt))

        Action_id = list([id_opt,index_opt[id_opt]])

    else:
        dist_id = 1  # dummy value
        for id_1 in range(0, action_space.size):
            k = 0
            X_old1 = Buffer[k]
            X_new, R1 = environment.environment(X_old1, car_id, id_1, t_step_DT, params, dist_id, Level_ratio)
            R1_max = max(R1_max, R1)
            if R1 < R1_max + dR_drop:
                continue

            Buffer[k + 1] = X_pseudo1[k + 1]
            Buffer[k + 1][:, car_id] = X_new[:, car_id]

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