#
# I N F O  5 2 9
#
# M I D T E R M  P R O J E C T
#
# This will predict crop yield given a dataset
#


# Toss aside warnings about future versions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import time
import logging
import tensorflow as tf
import argparse

DAYS = 214
FIRST_YEAR = 2003 # This was 1980 in the example
LAST_YEAR = 2015
NUM_YEARS = 13 # There are 13 years of data

# The number of weather samples
WEATHER = 7*DAYS  # 6 samples for 324 days = 1498
GENOTYPE = 1   # Genotype cluster
MISC = 3        # Location, Year, Yield


def conv_res_part_P(P_t,f,is_training,var_name):
    """
    Placeholder
    :param P_t:
    :param f: not used -- delete
    :param is_training: not used -- delete
    :param var_name: not used -- delete
    :return:
    """
    X=tf.contrib.layers.flatten(P_t)

    #print('conv2 out P', X)

    return X



def conv_res_part_E(E_t,f,is_training,var_name):
    """
    Placeholder
    :param E_t:
    :param f: not used -- delete
    :param is_training: not used -- delete
    :param var_name:
    :return:
    """
    s0 = 1

    X = tf.layers.conv1d(E_t, filters=8, kernel_size=9, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv00' + var_name, data_format='channels_last', reuse=tf.AUTO_REUSE)

    X = tf.nn.relu(X)
    #print('conv1 out E', X)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')

    X = tf.layers.conv1d(X, filters=12, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv0' + var_name, data_format='channels_last', reuse=tf.AUTO_REUSE)

    X = tf.nn.relu(X)

    #print('conv2 out E', X)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')

    X = tf.layers.conv1d(X, filters=16, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv1'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)




    X = tf.nn.relu(X)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')
    #print('conv3 out E',X)


    X = tf.layers.conv1d(X, filters=20, kernel_size=3, strides=s0, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv2'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')

    #print('conv4',X)

    return X




def conv_res_part_S(S_t,f,is_training,var_name):
    """
    Placeholder
    :param S_t:
    :param f:  not used -- delete
    :param is_training: not used -- delete
    :param var_name:
    :return:
    """
    X = tf.layers.conv1d(S_t, filters=4, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv1'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)



    X = tf.nn.relu(X)

    #print('conv1 out S',X)

    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')

    X = tf.layers.conv1d(X, filters=8, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv2'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)


    X = tf.nn.relu(X)

    #print('conv2 out S', X)

    X = tf.layers.conv1d(X, filters=12, kernel_size=2, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv3'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)


    X = tf.nn.relu(X)


    #print('conv3 out S', X)


    return X





# bem original
#def main_proccess(E_t1,E_t2,E_t3,E_t4,E_t5,E_t6,S_t1,S_t2,S_t3,S_t4,S_t5,S_t6,S_t7,S_t8,S_t9,S_t10,P_t,Ybar,S_t_extra,f,is_training,num_units,num_layers,dropout):
def main_process(E_t1, E_t2, E_t3, E_t4, E_t5, E_t6, E_t7, Ybar, f, is_training, num_units, num_layers, dropout):
    """
    Placeholder
    :param E_t1:
    :param E_t2:
    :param E_t3:
    :param E_t4:
    :param E_t5:
    :param E_t6:
    :param E_t7:
    :param Ybar:
    :param f:
    :param is_training:
    :param num_units:
    :param num_layers:
    :param dropout:
    :return:
    """
    e_out1 = conv_res_part_E(E_t1, f, is_training=is_training,var_name='v1')
    e_out1 = tf.contrib.layers.flatten(e_out1)
    e_out2 = conv_res_part_E(E_t2, f, is_training=is_training, var_name='v1')
    e_out2 = tf.contrib.layers.flatten(e_out2)
    e_out3 = conv_res_part_E(E_t3, f, is_training=is_training, var_name='v1')
    e_out3 = tf.contrib.layers.flatten(e_out3)
    e_out4 = conv_res_part_E(E_t4, f, is_training=is_training, var_name='v1')
    e_out4 = tf.contrib.layers.flatten(e_out4)
    e_out5 = conv_res_part_E(E_t5, f, is_training=is_training, var_name='v1')
    e_out5 = tf.contrib.layers.flatten(e_out5)
    e_out6 = conv_res_part_E(E_t6, f, is_training=is_training, var_name='v1')
    e_out6 = tf.contrib.layers.flatten(e_out6)



    e_out=tf.concat([e_out1,e_out2,e_out3,e_out4,e_out5,e_out6],axis=1)
    #print('after concatenate',e_out)

    e_out = tf.contrib.layers.fully_connected(inputs=e_out, num_outputs=60, activation_fn=tf.nn.relu,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer())


    # s_out1 = conv_res_part_S(S_t1, f, is_training=is_training, var_name='v1S')
    # s_out1 = tf.contrib.layers.flatten(s_out1)
    # s_out2 = conv_res_part_S(S_t2, f, is_training=is_training, var_name='v1S')
    # s_out2 = tf.contrib.layers.flatten(s_out2)
    # s_out3 = conv_res_part_S(S_t3, f, is_training=is_training, var_name='v1S')
    # s_out3 = tf.contrib.layers.flatten(s_out3)
    # s_out4 = conv_res_part_S(S_t4, f, is_training=is_training, var_name='v1S')
    # s_out4 = tf.contrib.layers.flatten(s_out4)
    # s_out5 = conv_res_part_S(S_t5, f, is_training=is_training, var_name='v1S')
    # s_out5 = tf.contrib.layers.flatten(s_out5)
    # s_out6 = conv_res_part_S(S_t6, f, is_training=is_training, var_name='v1S')
    # s_out6 = tf.contrib.layers.flatten(s_out6)

    #s_out7 = conv_res_part_S(S_t7, f, is_training=is_training, var_name='v1S')
    #s_out7 = tf.contrib.layers.flatten(s_out7)
    # bem comment out these lines begin
    # s_out7 = conv_res_part_S(S_t7, f, is_training=is_training, var_name='v1S')
    # s_out7 = tf.contrib.layers.flatten(s_out7)
    # s_out8 = conv_res_part_S(S_t8, f, is_training=is_training, var_name='v1S')
    # s_out8 = tf.contrib.layers.flatten(s_out8)
    # s_out9 = conv_res_part_S(S_t9, f, is_training=is_training, var_name='v1S')
    # s_out9 = tf.contrib.layers.flatten(s_out9)
    # s_out10 = conv_res_part_S(S_t10, f, is_training=is_training, var_name='v1S')
    # s_out10 = tf.contrib.layers.flatten(s_out10)
    # bem comment out these lines end

    #p_out=conv_res_part_P(P_t,f,is_training,var_name='P')
    #p_out=tf.contrib.layers.flatten(p_out)

    #print('p outtttttt',p_out)
    # print('E output----', e_out1)
    # e_out = tf.contrib.layers.flatten(e_out)

    # SOIL
    #s_out = tf.concat([s_out1, s_out2, s_out3, s_out4, s_out5, s_out6,s_out7,s_out8,s_out9,s_out10], axis=1)
    #s_out = tf.concat([s_out1, s_out2, s_out3, s_out4, s_out5, s_out6], axis=1)
    #print('soil after concatenate', s_out)

    # s_out = tf.contrib.layers.fully_connected(inputs=s_out, num_outputs=40, activation_fn=None,
    #                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
    #                                            biases_initializer=tf.zeros_initializer())



    #s_out = tf.nn.relu(s_out)


    #print('soil after FC layer', s_out)
    #print(s_out,'*****S_out')
    #print(e_out,'*****e_out')
    #print(p_out,'******p_out')
    # bem original: e_out=tf.concat([e_out,s_out,p_out],axis=1)
    # As p_out is not present, this step is probably not required
    #e_out=tf.concat([e_out,p_out],axis=1)

    #print('soil + Weather after concatante', e_out)

    time_step=5

    #print(e_out,'e_out1111111')
    e_out=tf.reshape(e_out,shape=[-1,time_step,e_out.get_shape().as_list()[-1]])

    #print('e_out_after_reshapeeeee',e_out)
    # bem comment out
    # S_t_extra=tf.reshape(S_t_extra,shape=[-1,time_step,4])
    e_out=tf.concat([e_out,Ybar],axis=-1)
    # bem comment out -- a bit inefficient with the above line as well
    #e_out = tf.concat([e_out, Ybar,S_t_extra], axis=-1)

    cells = []

    for _ in range(num_layers):
        #cell = tf.contrib.rnn.LSTMCell(num_units)

        cell = tf.contrib.rnn.LSTMCell(num_units)

        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)

        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)



    output, _= tf.nn.dynamic_rnn(cell, e_out, dtype=tf.float32)

    print('RNN output',output)



    output=tf.reshape(output,shape=[-1,output.get_shape().as_list()[-1]])

    print('my RNN output before FC layer',output)
    output = tf.contrib.layers.fully_connected(inputs=output, num_outputs=1, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())

    print(output)

    output = tf.reshape(output, shape=[-1,5])
    print("output of all time steps", output)
    Yhat1 = tf.gather(output, indices=[4], axis=1)

    print('Yhat1', Yhat1)

    Yhat2 = tf.gather(output, indices=[0,1,2,3], axis=1)
    print('Yhat2', Yhat2)

    return Yhat1,Yhat2



def Cost_function(Y, Yhat):
    """
    Placeholder
    :param Y:
    :param Yhat:
    :return:
    """
    E = Y - Yhat
    E2 = tf.pow(E, 2)

    MSE = tf.squeeze(tf.reduce_mean(E2))
    RMSE = tf.pow(MSE, 0.5)
    Loss = tf.losses.huber_loss(Y, Yhat, weights=1.0, delta=5.0)

    return RMSE, MSE, E, Loss





def get_sample(dic,L,avg,batch_size,time_steps,num_features):
    """
    Placeholder
    :param dic:
    :param L:
    :param avg:
    :param batch_size:
    :param time_steps:
    :param num_features:
    :return:
    """
    L_tr=L[:-1,:]

    out=np.zeros(shape=[batch_size,time_steps,num_features])

    for i in range(batch_size):

        r1 = np.squeeze(np.random.randint(L_tr.shape[0], size=1))

        years = L_tr[r1, :]

        for j, y in enumerate(years):
            X = dic[str(y)]
            ym=avg[str(y)]
            r2 = np.random.randint(X.shape[0], size=1)
            #n=X[r2, :]
            out[i, j, :] = np.concatenate((X[r2, :],np.array([[ym]])),axis=1)

    return out



def get_sample_te(dic,mean_last,avg,batch_size_te,time_steps,num_features):
    """
    Placeholder
    :param dic:
    :param mean_last:
    :param avg:
    :param batch_size_te:
    :param time_steps:
    :param num_features:
    :return:
    """
    out = np.zeros(shape=[batch_size_te, time_steps, num_features])

    X = dic[str(LAST_YEAR)]

    #  r1 = np.random.randint(X.shape[0], size=batch_size_te)
    # bem original
    #out[:, 0:4, :] += mean_last.reshape(1,4,3+6*52+1+100+16+4)
    # bem - where did this 3+ xxx and 1 + xxx come from?
    out[:, 0:4, :] += mean_last.reshape(1,4,WEATHER + MISC + 1)
    #n=X[r1, :]
    #print(n.shape)
    ym=np.zeros(shape=[batch_size_te,1])+avg[str(LAST_YEAR)]

    out[:,4,:]=np.concatenate((X,ym),axis=1)

    return out







def main_program(X, Index,num_units,num_layers,Max_it, learning_rate, batch_size_tr,le,l):
    """
    Placeholder
    :param X:
    :param Index:
    :param num_units:
    :param num_layers:
    :param Max_it:
    :param learning_rate:
    :param batch_size_tr:
    :param le:
    :param l:
    :return:
    """
    with tf.device('/cpu:0'):

        # Weather observations
        E_t1 = tf.placeholder(shape=[None, DAYS,1], dtype=tf.float32, name='E_t1')

        E_t2 = tf.placeholder(shape=[None, DAYS, 1], dtype=tf.float32, name='E_t2')

        E_t3 = tf.placeholder(shape=[None, DAYS, 1], dtype=tf.float32, name='E_t3')
        E_t4 = tf.placeholder(shape=[None, DAYS, 1], dtype=tf.float32, name='E_t4')
        E_t5 = tf.placeholder(shape=[None, DAYS, 1], dtype=tf.float32, name='E_t5')
        E_t6 = tf.placeholder(shape=[None, DAYS, 1], dtype=tf.float32, name='E_t6')
        E_t7 = tf.placeholder(shape=[None, DAYS, 1], dtype=tf.float32, name='E_t7')

        # bem S_t_extra=tf.placeholder(shape=[None, 4,1], dtype=tf.float32, name='S_t_extra')
        #bem end modifications

        #P_t = tf.placeholder(shape=[None, 16, 1], dtype=tf.float32, name='P_t')  #Plant Data

        Ybar=tf.placeholder(shape=[None,5,1], dtype=tf.float32, name='Ybar')

        Y_t = tf.placeholder(shape=[None, 1], dtype=tf.float32,name='Y_t')

        Y_t_2 = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='Y_t_2')

        is_training=tf.placeholder(dtype=tf.bool)
        lr=tf.placeholder(shape=[],dtype=tf.float32,name='learning_rate')
        dropout = tf.placeholder(tf.float32,name='dropout')

        f=3
        # bem original
        #Yhat1,Yhat2= main_proccess(E_t1,E_t2,E_t3,E_t4,E_t5,E_t6,S_t1,S_t2,S_t3,S_t4,S_t5,S_t6,S_t7,S_t8,S_t9,S_t10,P_t,Ybar,S_t_extra,f,is_training,num_units,num_layers,dropout)
        # bem reduced set
        Yhat1,Yhat2= main_process(E_t1, E_t2, E_t3, E_t4, E_t5, E_t6, E_t7, Ybar, f, is_training, num_units, num_layers, dropout)
        Yhat1=tf.identity(Yhat1,name='Yhat1')
        # Yhat2 is the prediction we got before the final time step (year t)
        print('Yhat1',Yhat1)

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            #print(variable)
            shape = variable.get_shape()
            #print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                #   print(dim)
                variable_parameters *= dim.value
            print("Variable parameters: {}".format(variable_parameters))
            total_parameters += variable_parameters
        print("total_parameters",total_parameters)

        with tf.name_scope('loss_function'):

            RMSE,_,_,Loss1=Cost_function(Y_t, Yhat1)

            _, _, _, Loss2 = Cost_function(Y_t_2, Yhat2)

            Tloss=tf.constant(l,dtype=tf.float32)*Loss1+tf.constant(le,dtype=tf.float32)*Loss2

        RMSE=tf.identity(RMSE,name='RMSE')
        with tf.name_scope('train'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(Tloss)


        init = tf.global_variables_initializer()

        sess = tf.Session()

        sess.run(init)
        writer = tf.summary.FileWriter("./tensorboard")
        writer.add_graph(sess.graph)

        t1=time.time()

        A = []

        # Original: for i in range(4, 39):
        for i in range(MISC + 1, NUM_YEARS + 1):
            A.append([ i - 4, i - 3, i - 2, i - 1, i])

        A = np.vstack(A)
        A += FIRST_YEAR
        #print(A.shape)

        dic = {}

        # number of years
        for i in range(NUM_YEARS + 1):
            # cast this to float?
            dic[str(i + FIRST_YEAR)] = X[X[:, 1] == float(i + FIRST_YEAR)]


        avg = {}
        avg2 = []
        for i in range(NUM_YEARS):
            avg[str(i + FIRST_YEAR)] = np.mean(X[X[:, 1] == i + FIRST_YEAR][:, 2])
            avg2.append(np.mean(X[X[:, 1] == i + FIRST_YEAR][:, 2]))

        #print('avgggggg', avg)

        mm = np.mean(avg2)
        ss = np.std(avg2)


        avg = {}

        for i in range(NUM_YEARS):
            avg[str(i + FIRST_YEAR)] = (np.mean(X[X[:, 1] == i + FIRST_YEAR][:, 2]) - mm) / ss

        avg['2015'] = avg['2014']
        #avg['2017']=avg['2016']


        #a2 = np.concatenate((np.mean(dic['2008'], axis=0), [avg['2008']]))

        #a3 = np.concatenate((np.mean(dic['2009'], axis=0), [avg['2009']]))

        #a4 = np.concatenate((np.mean(dic['2010'], axis=0), [avg['2010']]))

        #a5 = np.concatenate((np.mean(dic['2011'], axis=0), [avg['2011']]))

        #a6 = np.concatenate((np.mean(dic['2012'], axis=0), [avg['2012']]))

        #a7 = np.concatenate((np.mean(dic['2013'], axis=0), [avg['2013']]))

        a8 = np.concatenate((np.mean(dic['2012'], axis=0), [avg['2012']]))

        a9 = np.concatenate((np.mean(dic['2013'], axis=0), [avg['2013']]))
        a10 = np.concatenate((np.mean(dic['2014'], axis=0), [avg['2014']]))

        a11 = np.concatenate((np.mean(dic['2015'], axis=0), [avg['2015']]))

        mean_last = np.concatenate((a8, a9, a10,a11))

        loss_validation=[]

        loss_train=[]

        for i in range(Max_it):

            #out_tr = get_sample(dic, A, avg,batch_size_tr, time_steps=5, num_features=316+100+16+4)
            # bem: test number of features as 6*52=312 (weather) 6*11 (soil), 16 (planting) 3 (loc, year, yield)
            # why does timesteps - 5?  We have only 4 years, right?
            #WEATHER = 312
            #SOIL = 66
            #PLANTING = 16
            #MISC = 3
            out_tr = get_sample(dic, A, avg,batch_size_tr, time_steps=5, num_features=WEATHER + MISC + 1)

            Ybar_tr=out_tr[:, :, -1].reshape(-1,5,1)

            #Batch_X_e = out_tr[:, :, 3:-1].reshape(-1,6*52+100+16+4)
            # bem: same thing here as above.
            out = out_tr[:, :, 3:-1]
            Batch_X_e = out_tr[:, :, 3:-1].reshape(-1,WEATHER)


            Batch_X_e=np.expand_dims(Batch_X_e,axis=-1)
            Batch_Y = out_tr[:, -1, 2]
            Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)

            Batch_Y_2 = out_tr[:, np.arange(0,4), 2]


            if i==60000:
                learning_rate=learning_rate/2
                print('learningrate1',learning_rate)
            elif i==120000:
                learning_rate = learning_rate/2
                print('learningrate2', learning_rate)
            elif i==180000:
                learning_rate = learning_rate/2
                print('learningrate3', learning_rate)


            # bem: Hmm Batch_X_e is (80,394,1), so this can't really work.  No wonder it craps out on t9 below.
            # Originally
            # sess.run(train_op, feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
            #                                E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
            #                                S_t1:Batch_X_e[:,312:322,:],S_t2:Batch_X_e[:,322:332,:],S_t3:Batch_X_e[:,332:342,:],
            #                                S_t4:Batch_X_e[:,342:352,:],S_t5:Batch_X_e[:,352:362,:],S_t6:Batch_X_e[:,362:372,:],
            #                                S_t7: Batch_X_e[:, 372:382, :], S_t8: Batch_X_e[:, 382:392, :],
            #                                S_t9: Batch_X_e[:, 392:402, :], S_t10: Batch_X_e[:, 402:412, :],P_t: Batch_X_e[:, 412:428, :],S_t_extra:Batch_X_e[:, 428:, :],
            #                                Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})
            sess.run(train_op, feed_dict={ E_t1: Batch_X_e[:,0:DAYS,:],E_t2: Batch_X_e[:,DAYS*1:2*DAYS,:],E_t3: Batch_X_e[:,DAYS*2:3*DAYS,:],
                                           E_t4: Batch_X_e[:, DAYS * 3:4 * DAYS, :],E_t5: Batch_X_e[:,DAYS*4:5*DAYS,:],E_t6: Batch_X_e[:,DAYS*5:DAYS*6,:], # End of weather readings
                                           #P_t: Batch_X_e[:, 378:394, :],#S_t_extra:Batch_X_e[:, 428:, :],
                                           Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})

            if i%1000==0:

                # bem Original
                #out_tr = get_sample(dic, A, avg, batch_size=1000, time_steps=5, num_features=316 + 100 + 16 + 4)
                out_tr = get_sample(dic, A, avg, batch_size=1000, time_steps=5, num_features=WEATHER + MISC + 1)

                Ybar_tr = out_tr[:, :, -1].reshape(-1, 5, 1)

                # bem original
                #Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 6 * 52 + 100 + 16 + 4)
                # bem -- not sure I can explain why I have to drop the size of this by 4
                Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, WEATHER)


                Batch_X_e = np.expand_dims(Batch_X_e, axis=-1)
                Batch_Y = out_tr[:, -1, 2]
                Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)

                Batch_Y_2 = out_tr[:, np.arange(0, 4), 2]

                # bem original
                #out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=316+100+16+4)
                out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=WEATHER + MISC + 1)
                print(out_te.shape)
                Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
                # bem original
                #Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*52+100+16+4)
                # bem -- not sure I can explain why I have to drop the size of this by 4
                Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,WEATHER)
                Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
                Batch_Y_te = out_te[:, -1, 2]
                Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
                Batch_Y_te2 = out_te[:, np.arange(0,4), 2]

                # bem originally
                # rmse_tr,yhat1_tr,loss_tr = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
                #                                                                     E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
                #                                                                     S_t1:Batch_X_e[:,312:322,:],S_t2:Batch_X_e[:,322:332,:],S_t3:Batch_X_e[:,332:342,:],
                #                                                                     S_t4:Batch_X_e[:,342:352,:],S_t5:Batch_X_e[:,352:362,:],S_t6:Batch_X_e[:,362:372,:],
                #                                                                     # bem commented out
                #                                                                     #S_t7: Batch_X_e[:, 372:382, :], S_t8: Batch_X_e[:, 382:392, :],
                #                                                                     #S_t9: Batch_X_e[:, 392:402, :], S_t10: Batch_X_e[:, 402:412, :],P_t: Batch_X_e[:, 412:428, :],S_t_extra:Batch_X_e[:, 428:, :],
                #                                                                     Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})
                rmse_tr,yhat1_tr,loss_tr = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e[:,0:DAYS,:],E_t2: Batch_X_e[:,DAYS*1:2*DAYS,:],E_t3: Batch_X_e[:,DAYS*2:3*DAYS,:],
                                           E_t4: Batch_X_e[:, DAYS * 3:4 * DAYS, :],E_t5: Batch_X_e[:,DAYS*4:5*DAYS,:],E_t6: Batch_X_e[:,DAYS*5:DAYS*6,:],
                                           #P_t: Batch_X_e[:, 378:394, :], #S_t_extra:Batch_X_e[:, 428:, :],
                                           Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})

                rc_tr = np.corrcoef(np.squeeze(Batch_Y), np.squeeze(yhat1_tr))[0, 1]

                # bem originally
                # rmse_te,yhat1_te,loss_val = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e_te[:,0:DAYS,:],E_t2: Batch_X_e_te[:,DAYS*1:2*DAYS,:],E_t3: Batch_X_e_te[:,DAYS*2:3*DAYS,:],
                #                                                                      E_t4: Batch_X_e_te[:, DAYS * 3:4 * DAYS, :],E_t5: Batch_X_e_te[:,DAYS*4:5*DAYS,:],E_t6: Batch_X_e_te[:,DAYS*5:DAYS*6,:],
                #                                                                      S_t1:Batch_X_e_te[:,312:322,:],S_t2:Batch_X_e_te[:,322:332,:],S_t3:Batch_X_e_te[:,332:342,:],
                #                                                                      S_t4:Batch_X_e_te[:,342:3DAYS,:],S_t5:Batch_X_e_te[:,3DAYS:362,:],S_t6:Batch_X_e_te[:,362:372,:],
                #                                                                      # bem commented out
                #                                                                      #S_t7: Batch_X_e_te[:, 372:382, :], S_t8: Batch_X_e_te[:, 382:392, :],
                #                                                                      #S_t9: Batch_X_e_te[:, 392:402, :], S_t10: Batch_X_e_te[:, 402:412, :],
                #                                                                      P_t: Batch_X_e_te[:, 412:428, :], # S_t_extra:Batch_X_e_te[:, 428:, :],
                #                                                                      Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})

                rmse_te,yhat1_te,loss_val = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e_te[:,0:DAYS,:],E_t2: Batch_X_e_te[:,DAYS*1:2*DAYS,:],E_t3: Batch_X_e_te[:,DAYS*2:3*DAYS,:],
                                           E_t4: Batch_X_e_te[:, DAYS * 3:4 * DAYS, :],E_t5: Batch_X_e_te[:,DAYS*4:5*DAYS,:],E_t6: Batch_X_e_te[:,DAYS*5:DAYS*6,:],
                                           #P_t: Batch_X_e_te[:, 378:394, :], # S_t_extra:Batch_X_e_te[:, 428:, :],
                                           Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})

                rc=np.corrcoef(np.squeeze(Batch_Y_te),np.squeeze(yhat1_te))[0,1]
                loss_validation.append(loss_val)

                loss_train.append(loss_tr)

                print(loss_tr,loss_val)
                print("Iteration %d , The training RMSE is %f and Cor train is %f  and test RMSE is %f and Cor is %f " % (i, rmse_tr,rc_tr, rmse_te,rc))


    # bem original
    #out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=316+100+16+4)
    out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=WEATHER + MISC + 1)

    # bem original
    #Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*DAYS+100+16+4)
    # bem not sure why i need to drop this size by 4
    Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,WEATHER)
    Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
    Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
    Batch_Y_te = out_te[:, -1, 2]
    Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
    Batch_Y_te2 = out_te[:, np.arange(0, 4), 2]

    # bem original
    # rmse_te,yhat1 = sess.run([RMSE,Yhat1], feed_dict={ E_t1: Batch_X_e_te[:,0:DAYS,:],E_t2: Batch_X_e_te[:,DAYS*1:2*DAYS,:],E_t3: Batch_X_e_te[:,DAYS*2:3*DAYS,:],
    #                                                    E_t4: Batch_X_e_te[:, DAYS * 3:4 * DAYS, :],E_t5: Batch_X_e_te[:,DAYS*4:5*DAYS,:],E_t6: Batch_X_e_te[:,DAYS*5:DAYS*6,:],
    #                                                    S_t1:Batch_X_e_te[:,312:322,:],S_t2:Batch_X_e_te[:,322:332,:],S_t3:Batch_X_e_te[:,332:342,:],
    #                                                    S_t4:Batch_X_e_te[:,342:3DAYS,:],S_t5:Batch_X_e_te[:,3DAYS:362,:],S_t6:Batch_X_e_te[:,362:372,:],
    #                                                    S_t7: Batch_X_e_te[:, 372:382, :], S_t8: Batch_X_e_te[:, 382:392, :],
    #                                                    S_t9: Batch_X_e_te[:, 392:402, :], S_t10: Batch_X_e_te[:, 402:412, :],
    #                                                    P_t: Batch_X_e_te[:, 412:428, :],#S_t_extra:Batch_X_e_te[:, 428:, :],
    #                                                    Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})
    rmse_te,yhat1 = sess.run([RMSE,Yhat1], feed_dict={ E_t1: Batch_X_e_te[:,0:DAYS,:],E_t2: Batch_X_e_te[:,DAYS*1:2*DAYS,:],E_t3: Batch_X_e_te[:,DAYS*2:3*DAYS,:],
                                           E_t4: Batch_X_e_te[:, DAYS * 3:4 * DAYS, :],E_t5: Batch_X_e_te[:,DAYS*4:5*DAYS,:],E_t6: Batch_X_e_te[:,DAYS*5:DAYS*6,:],
                                           #P_t: Batch_X_e_te[:, 378:394, :],#S_t_extra:Batch_X_e_te[:, 428:, :],
                                           Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})


    print("The training RMSE is %f  and test RMSE is %f " % (rmse_tr, rmse_te))
    t2=time.time()

    print('the training time was %f' %(round(t2-t1,2)))
    saver = tf.train.Saver()
    #saver.save(sess, './model_corn', global_step=i)  # Saving the model
    saver.save(sess, os.path.join(os.getcwd(), arguments.path, arguments.model + '.ckpt'))
    #saver.save(sess, arguments.model, global_step=i)  # Saving the model

    return  rmse_tr,rmse_te,loss_train,loss_validation



Max_it=350000      #150000 could also be used
learning_rate=0.0003   # Learning rate
batch_size_tr=16  # traning batch size
le=0.0  # Weight of loss for prediction using times before final time steps
l=1.0    # Weight of loss for prediction using final time step
num_units=64  # Number of hidden units for LSTM celss
num_layers=1  # Number of layers of LSTM cell

parser = argparse.ArgumentParser("Yield prediction")

parser.add_argument('-d', '--data', action="store", required=True, help="Combined data")
parser.add_argument('-i', '--iterations', action="store", required=True, type=int, help="Number of iterations")
parser.add_argument('-m', '--model', action="store", required=True, type=str, help="Model Name")
parser.add_argument('-p', '--path', action="store", required=True, type=str, help="Model path")
parser.add_argument('-l', '--learning', action="store", required=False, type=float, default=learning_rate, help="Learning rate")
parser.add_argument('-b', '--batch', action="store", required=False, type=int, default=batch_size_tr, help="Batch size")
parser.add_argument('-u', '--units', action="store", required=False, type=int, default=num_units, help="Hidden units for LSTM cells")
parser.add_argument('-t', '--thickness', action="store", required=False, type=int, default=num_layers, help="Layers of LSTM cells")
parser.add_argument('-wb', '--weight_before', action="store", required=False, type=float, default=le, help="Weight of loss before final time steps")
parser.add_argument('-wf', '--weight_final', action="store", required=False, type=float, default=l, help="Weight pf loss using final step")

arguments = parser.parse_args()

import sys
sys.stderr.write("Training")
sys.stderr.flush()
# Turn off annoying debug messages
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#BigX = np.load('./corn.npz') ##order W(DAYS*6) S(100) P(16) S_extra(4)
BigX = np.load(arguments.data) ##order W(DAYS*6) S(100) P(16) S_extra(4)

X=BigX['data']

X_tr=X[X[:,1]<=2017]

X_tr=X_tr[:,3:]

M=np.mean(X_tr,axis=0,keepdims=True)
S=np.std(X_tr,axis=0,keepdims=True)
X[:,3:]=(X[:,3:]-M)/S

index_low_yield=X[:,2]<10
print('low yield observations',np.sum(index_low_yield))
print(X[index_low_yield][:,1])
X=np.nan_to_num(X)
X=X[np.logical_not(index_low_yield)]

del BigX

Index=X[:,1]==LAST_YEAR  #validation year

print("train data",np.sum(np.logical_not(Index)))
print("test data",np.sum(Index))

print('Std %.2f and mean %.2f  of test ' %(np.std(X[Index][:,2]),np.mean(X[Index][:,2])) , flush=true)

Max_it=350000      #150000 could also be used
learning_rate=0.0003   # Learning rate
batch_size_tr=16  # traning batch size
le=0.0  # Weight of loss for prediction using times before final time steps
l=1.0    # Weight of loss for prediction using final time step
num_units=64  # Number of hidden units for LSTM celss
num_layers=1  # Number of layers of LSTM cell

# bem original
#rmse_tr,rmse_te,train_loss,validation_loss=main_program(X, Index,num_units,num_layers,Max_it, learning_rate, batch_size_tr,le,l)
rmse_tr,rmse_te,train_loss,validation_loss=main_program(X,
                                                        Index,
                                                        arguments.units,
                                                        arguments.thickness,
                                                        arguments.iterations,
                                                        arguments.learning,
                                                        arguments.batch,
                                                        arguments.weight_before,
                                                        arguments.weight_final)




