# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os
import sys
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, ConcatLayer, SliceLayer
from generators import ModularCueCombinationTaskFFWD
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
    l_in_hid_1   = DenseLayer( SliceLayer(l_in,indices=slice(None,2*(n_in/3)), axis=1), n_hid, nonlinearity=lasagne.nonlinearities.rectify)
    l_in_hid     = DenseLayer( ConcatLayer( (l_in_hid_1, SliceLayer(l_in,indices=slice(2*(n_in/3),None), axis=1) ), axis=1 ), n_hid, nonlinearity=lasagne.nonlinearities.rectify)
    
    # Output Layer
    l_shp        = ReshapeLayer(l_in_hid, (-1, n_hid))
    l_dense      = DenseLayer(l_shp, num_units=n_out, nonlinearity=lasagne.nonlinearities.linear)
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out        = ReshapeLayer(l_dense, (batch_size, n_out))

    return l_out, l_in_hid, l_in_hid_1

if __name__ == '__main__':
    # Define the input and expected output variable
    input_var, target_var = T.fmatrices('input', 'target')
    # The generator to sample examples from
    tr_cond               = 'two_gains'
    test_cond             = 'all_gains'
    generator             = ModularCueCombinationTaskFFWD(max_iter=50001, batch_size=100, n_in=50, n_out=1, sigma_sq=100.0, tr_cond=tr_cond)
    test_generator        = ModularCueCombinationTaskFFWD(max_iter=2501,  batch_size=100, n_in=50, n_out=1, sigma_sq=100.0, tr_cond=test_cond)

    # The model 
    l_out, l_rec, l_fix   = model(input_var, batch_size=generator.batch_size, n_in=3*generator.n_in, n_out=generator.n_out, n_hid=200)
    # The generated output variable and the loss function
#    all_layers            = lasagne.layers.get_all_layers(l_out)
#    l2_penalty            = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 1e-6
    pred_var              = lasagne.layers.get_output(l_out)
    loss                  = T.mean(lasagne.objectives.squared_error(pred_var, target_var)) # + l2_penalty
    
    # Create the update expressions
    params                = lasagne.layers.get_all_params(l_out, trainable=True)
    trainable_params      = params[2:]
    updates               = lasagne.updates.adam(loss, trainable_params, learning_rate=0.0003)
    
    # Compile the function for a training step, as well as the prediction function and a utility function to get the inner details of the RNN
    train_fn              = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    pred_fn               = theano.function([input_var], pred_var, allow_input_downcast=True)
    rec_layer_fn          = theano.function([input_var], lasagne.layers.get_output(l_rec, get_details=True), allow_input_downcast=True)

    # Set the untrained params
    lasagne.layers.set_all_param_values(l_fix, [sio.loadmat('ct_abserr_everything_allgains_run3.mat')['W_hid'].astype('float32'), 
                                                sio.loadmat('ct_abserr_everything_allgains_run3.mat')['b_hid'].flatten().astype('float32')])
    
    # TRAINING
    s_vec, opt_s_vec, ex_pred_vec, frac_rmse_vec = [], [], [], []
    for i, (example_input, example_output, g1, g2, s, opt_s) in generator:
        score              = train_fn(example_input, example_output)
        example_prediction = pred_fn(example_input)
        s_vec.append(s)
        opt_s_vec.append(opt_s)
        ex_pred_vec.append(example_prediction.squeeze())
        if i % 500 == 0:
            rmse_opt  = np.sqrt(np.nanmean((np.asarray(s_vec) - np.asarray(opt_s_vec))**2))
            rmse_net  = np.sqrt(np.nanmean((np.asarray(s_vec) - np.asarray(ex_pred_vec))**2))
            frac_rmse = (rmse_net - rmse_opt) / rmse_opt
            frac_rmse_vec.append(frac_rmse)
            print 'Batch #%d; Frac. RMSE: %.6f; Opt. RMSE: %.6f; Net. RMSE: %.6f' % (i, frac_rmse, rmse_opt, rmse_net)
            s_vec       = []
            opt_s_vec   = []
            ex_pred_vec = []
    
    # TESTING
    s_vec, opt_s_vec, ex_pred_vec = [], [], []
    for i, (example_input, example_output, g1, g2, s, opt_s) in test_generator:
        example_prediction = pred_fn(example_input)
        s_vec.append(s)
        opt_s_vec.append(opt_s)
        ex_pred_vec.append(example_prediction.squeeze())
        
    rmse_opt       = np.sqrt(np.nanmean((np.asarray(s_vec) - np.asarray(opt_s_vec))**2))
    rmse_net       = np.sqrt(np.nanmean((np.asarray(s_vec) - np.asarray(ex_pred_vec))**2))
    frac_rmse_test = (rmse_net - rmse_opt) / rmse_opt
    print 'Test data; Frac. RMSE: %.6f; Opt. RMSE: %.6f; Net. RMSE: %.6f' % (frac_rmse_test, rmse_opt, rmse_net)

    # SAVE TRAINED MODEL  
    sio.savemat('ctcc_abserr_testfrmse_twogains_run%i.mat'%job_idx, {'frac_rmse_vec':np.asarray(frac_rmse_vec), 'frac_rmse_test':frac_rmse_test, 'opt_vec':np.asarray(opt_s_vec), 'net_vec':np.asarray(ex_pred_vec) } )      
