#generates next-state from action and observation
def state_generator(action, obs):
    input_vector = []
    if action is None:
        logger.info(f'no action, hence, no next_state !')
        sys.exit()
    logger.info(f'action.size:{action.size}')
    for user_i in range(action.size):
        logger.info(f'user_i:{user_i} action:{action[user_i]}')
        input_vector_i = one_hot(action[user_i],NUM_CHANNELS+1)
        channel_alloc = obs[-1] # obs 뒤에서 첫번째
        input_vector_i = np.append(input_vector_i,channel_alloc)
        input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #ACK
        input_vector.append(input_vector_i)

    logger.error(f'@ state_generator - input_vector:{input_vector}')
    return input_vector