# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os
import sys
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import InputLayer, ReshapeLayer, SliceLayer, ElemwiseSumLayer, NonlinearityLayer, DenseLayer
from CellTypeSpecificLayers import DenseEILayer
from generators import ModularCueCombinationTask
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

def relu_non(x):
    return T.maximum(0.0,x)

def model(input_var, batch_size=1, n_in=100, n_out=1, n_hid=200, ei_ratio=0.8):
    # Input Layer
    l_in         = InputLayer((batch_size, None, n_in), input_var=input_var)
    l_in_1       = SliceLayer(l_in, indices=slice(None,2*(n_in/3)), axis=2 )
    l_in_2       = SliceLayer(l_in, indices=slice(2*(n_in/3),None), axis=2 )
    _, seqlen, _ = l_in.input_var.shape
    # Recurrent EI Net
    l_in_hid_1   = DenseLayer(lasagne.layers.InputLayer((None, 2*(n_in/3))), n_hid, W=lasagne.init.GlorotNormal(0.1), b=None, nonlinearity=lasagne.nonlinearities.linear)
    l_hid_hid_1  = DenseLayer(lasagne.layers.InputLayer((None, n_hid)), n_hid, W=lasagne.init.GlorotNormal(0.1), nonlinearity=lasagne.nonlinearities.linear)
    l_rec_1      = lasagne.layers.CustomRecurrentLayer( l_in_1, l_in_hid_1, l_hid_hid_1, nonlinearity=lasagne.nonlinearities.rectify)
    
    l_in_hid_2a  = DenseEILayer(lasagne.layers.InputLayer((None, n_hid)), ei_ratio, n_hid, W=lasagne.init.GlorotNormal(0.1), b=None, nonlinearity=lasagne.nonlinearities.linear, leak=1.0, diagonal=True)
    l_in_hid_2b  = DenseEILayer(lasagne.layers.InputLayer((None, n_in/3)), ei_ratio, n_hid, W=lasagne.init.GlorotNormal(0.1), b=None, nonlinearity=lasagne.nonlinearities.linear, leak=1.0, diagonal=True)

    l_hid_hid_2  = DenseEILayer(lasagne.layers.InputLayer((None, n_hid)), ei_ratio, n_hid, W=lasagne.init.GlorotNormal(0.1), nonlinearity=lasagne.nonlinearities.linear, leak=1.0, diagonal=False)
    
    l_rec_2a     = lasagne.layers.CustomRecurrentLayer( l_rec_1, l_in_hid_2a, l_hid_hid_2, nonlinearity=lasagne.nonlinearities.linear)
    l_rec_2b     = lasagne.layers.CustomRecurrentLayer( l_in_2, l_in_hid_2b, l_hid_hid_2, nonlinearity=lasagne.nonlinearities.linear)
    l_rec_2      = NonlinearityLayer(ElemwiseSumLayer((l_rec_2a,l_rec_2b)),nonlinearity=relu_non)
          
    # Output Layer
    l_shp        = ReshapeLayer(l_rec_2, (-1, n_hid))
    l_dense      = DenseEILayer(l_shp, ei_ratio, num_units=n_out, W=lasagne.init.GlorotNormal(0.1), nonlinearity=lasagne.nonlinearities.linear, leak=1.0, diagonal=True)
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out        = ReshapeLayer(l_dense, (batch_size, seqlen, n_out))

    return l_out, l_rec_1, l_rec_2

if __name__ == '__main__':
    # Define the input and expected output variable
    input_var, target_var = T.tensor3s('input', 'target')
    
    # The generator to sample examples from
    tr_cond               = 'two_gains'
    test_cond             = 'all_gains'
    generator             = ModularCueCombinationTask(max_iter=50001, batch_size=100, n_in=50, n_out=1, stim_dur=10, sigma_sq=100.0, tr_cond=tr_cond)
    test_generator        = ModularCueCombinationTask(max_iter=2501, batch_size=100, n_in=50, n_out=1, stim_dur=10, sigma_sq=100.0, tr_cond=test_cond)

    # The model 
    l_out, l_fix, l_rec   = model(input_var, batch_size=generator.batch_size, n_in=3*generator.n_in, n_out=generator.n_out, n_hid=200, ei_ratio=0.8)
    # The generated output variable and the loss function
    pred_var              = lasagne.layers.get_output(l_out)
    loss                  = T.mean(lasagne.objectives.squared_error(pred_var[:,-1,-1], target_var[:,-1,-1]))
    
    # Create the update expressions
    params                = lasagne.layers.get_all_params(l_out, trainable=True)
    trainable_params      = params[3:]
    updates               = lasagne.updates.adam(loss, trainable_params, learning_rate=0.001)
    
    # Compile the function for a training step, as well as the prediction function and a utility function to get the inner details of the RNN
    train_fn              = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    pred_fn               = theano.function([input_var], pred_var, allow_input_downcast=True)

    # Set the untrained params
    lasagne.layers.set_all_param_values(l_fix, [np.zeros((1,200)).astype('float32'),
                                                sio.loadmat('ei_cc_everything_allgains_run7.mat')['W_in'].astype('float32'), 
                                                sio.loadmat('ei_cc_everything_allgains_run7.mat')['W_hid'].astype('float32'), 
                                                sio.loadmat('ei_cc_everything_allgains_run7.mat')['b_hid'].flatten().astype('float32')])
    
    # TRAINING
    s_vec, opt_s_vec, ex_pred_vec, frac_rmse_vec = [], [], [], []
    for i, (example_input, example_output, g1, g2, s, opt_s) in generator:
        score              = train_fn(example_input, example_output)
        example_prediction = pred_fn(example_input)
        s_vec.append(s)
        opt_s_vec.append(opt_s)
        ex_pred_vec.append(example_prediction[:,-1,-1])
        # plt.imshow(pred_fn_rec(example_input)[0,:,:],interpolation='nearest'); plt.gca().set_aspect('equal', adjustable='box'); plt.colorbar(); plt.show()
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
        ex_pred_vec.append(example_prediction[:,-1,-1])

    rmse_opt  = np.sqrt(np.nanmean((np.asarray(s_vec) - np.asarray(opt_s_vec))**2))
    rmse_net  = np.sqrt(np.nanmean((np.asarray(s_vec) - np.squeeze(np.asarray(ex_pred_vec)))**2))
    frac_rmse_test = (rmse_net - rmse_opt) / rmse_opt
    print 'Test data; Frac. RMSE: %.6f; Opt. RMSE: %.6f; Net. RMSE: %.6f' % (frac_rmse_test, rmse_opt, rmse_net)
    
    # SAVE TRAINED MODEL  
    sio.savemat('ei_cc_cc_testinfloss_twogains_run%i.mat'%job_idx, {'frac_rmse_test':frac_rmse_test, 'opt_vec':np.asarray(opt_s_vec), 'net_vec':np.asarray(ex_pred_vec) } )      