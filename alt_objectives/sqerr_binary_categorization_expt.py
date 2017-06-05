# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os
import sys
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
from generators import BinaryCategorizationTaskFFWD
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init
import scipy.io as sio

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
job_idx    = int(os.getenv('PBS_ARRAYID'))
np.random.seed(job_idx)

def model(input_var, batch_size=1, n_in=100, n_out=1, n_hid=200):

    # Input Layer
    l_in         = InputLayer((batch_size, n_in), input_var=input_var)
    # Hidden layer
    l_in_hid     = DenseLayer(l_in, n_hid, nonlinearity=lasagne.nonlinearities.rectify)
    # Output Layer
    l_shp        = ReshapeLayer(l_in_hid, (-1, n_hid))
    l_dense      = DenseLayer(l_shp, num_units=n_out, nonlinearity=lasagne.nonlinearities.sigmoid)
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out        = ReshapeLayer(l_dense, (batch_size, n_out))

    return l_out, l_in_hid

if __name__ == '__main__':
    # Define the input and expected output variable
    input_var, target_var = T.fmatrices ('input', 'target')
    
    # The generator to sample examples from
    tr_cond               = 'two_gains'
    test_cond             = 'all_gains'
    generator             = BinaryCategorizationTaskFFWD(max_iter=150001, batch_size=10, n_in=50, n_out=1, sigma_sq=100.0, tr_cond=tr_cond)
    test_generator        = BinaryCategorizationTaskFFWD(max_iter=2501,   batch_size=10, n_in=50, n_out=1, sigma_sq=100.0, tr_cond=test_cond)
    l_out, l_rec          = model(input_var, batch_size=generator.batch_size, n_in=generator.n_in, n_out=generator.n_out, n_hid=200)
    
    # The generated output variable and the loss function
#    all_layers            = lasagne.layers.get_all_layers(l_out)
#    l2_penalty            = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 1e-6
    pred_var              = T.clip(lasagne.layers.get_output(l_out), 1e-6, 1. - 1e-6)
    loss                  = T.mean(lasagne.objectives.squared_error(pred_var, target_var)) # + l2_penalty
    
    # Create the update expressions
    params                = lasagne.layers.get_all_params(l_out, trainable=True)
    updates               = lasagne.updates.adam(loss, params, learning_rate=0.0003)
    
    # Compile the function for a training step, as well as the prediction function and a utility function to get the inner details of the RNN
    train_fn              = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    pred_fn               = theano.function([input_var], pred_var, allow_input_downcast=True)
    rec_layer_fn          = theano.function([input_var], lasagne.layers.get_output(l_rec, get_details=True), allow_input_downcast=True)

    # If want to continue training an old model, uncomment below
#    npzfile_lout          = np.load('lout_trained_model.npz')
#    npzfile_lrec          = np.load('lrec_trained_model.npz')
#    lasagne.layers.set_all_param_values(l_out,[npzfile_lout['arr_0'],npzfile_lout['arr_1'],npzfile_lout['arr_2'],npzfile_lout['arr_3'],npzfile_lout['arr_4'],npzfile_lout['arr_5'],npzfile_lout['arr_6']])
#    lasagne.layers.set_all_param_values(l_rec,[npzfile_lout['arr_0'],npzfile_lout['arr_1'],npzfile_lout['arr_2'],npzfile_lout['arr_3'],npzfile_lout['arr_4']])

    # TRAINING
    opt_vec, net_vec, inf_loss_vec = [], [], []
    for i, (example_input, example_output, g, c, p) in generator:
        score              = train_fn(example_input, example_output)
        example_prediction = pred_fn(example_input)
        opt_vec.append(p)
        net_vec.append(example_prediction.squeeze())
        if i % 500 == 0:
            opt_vec  = np.asarray(opt_vec)
            net_vec  = np.asarray(net_vec)
            inf_loss = np.nanmean( opt_vec * np.log(opt_vec/net_vec) + (1.0 - opt_vec) * np.log((1.0 - opt_vec)/(1.0 - net_vec)) ) / np.nanmean( opt_vec * np.log(2.0*opt_vec) + (1.0-opt_vec) * np.log(2.0*(1.0-opt_vec)) ) 
            inf_loss_vec.append(inf_loss)
            print 'Batch #%d; Infloss: %.6f' % (i, inf_loss)
            opt_vec  = []
            net_vec  = []

    # TESTING
    opt_vec, net_vec, ex_hid_vec, ex_inp_vec = [], [], [], []
    for i, (example_input, example_output, g, c, p) in test_generator:
        example_prediction = pred_fn(example_input)
        example_hidden     = rec_layer_fn(example_input)
        opt_vec.append(p)
        net_vec.append(example_prediction.squeeze())
        if i % 10 == 0:
            ex_hid_vec.append(example_hidden)
            ex_inp_vec.append(example_input)

    opt_vec       = np.asarray(opt_vec)
    net_vec       = np.asarray(net_vec)
    inf_loss_test = np.nanmean( opt_vec * np.log(opt_vec/net_vec) + (1.0 - opt_vec) * np.log((1.0 - opt_vec)/(1.0 - net_vec)) ) / np.nanmean( opt_vec * np.log(2.0*opt_vec) + (1.0-opt_vec) * np.log(2.0*(1.0-opt_vec)) ) 
    print 'Test data; Infloss: %.6f'%inf_loss_test

    ex_hid_vec = np.asarray(ex_hid_vec)    
    ex_hid_vec = np.reshape(ex_hid_vec,(-1,200))    

    ex_inp_vec = np.asarray(ex_inp_vec)    
    ex_inp_vec = np.reshape(ex_inp_vec,(-1,50))    

    # SAVE TRAINED MODEL  
    sio.savemat('bc_sqerr_testinfloss_twogains_run%i.mat'%job_idx, {'inf_loss_test':inf_loss_test, 'opt_vec':opt_vec, 'net_vec':net_vec})      
#    sio.savemat('bc_everything_allgains_run%i.mat'%job_idx, {'W_hid': lasagne.layers.get_all_param_values(l_out)[0], 
#                                                             'b_hid': lasagne.layers.get_all_param_values(l_out)[1],
#                                                             'w_out': lasagne.layers.get_all_param_values(l_out)[2],
#                                                             'b_out': lasagne.layers.get_all_param_values(l_out)[3],
#                                                             'inf_loss_vec':np.asarray(inf_loss_vec), 
#                                                             'inpResps':ex_inp_vec, 
#                                                             'hidResps':ex_hid_vec} )               