# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:18 2016 by emin
"""
import os 
import sys
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import InputLayer, ReshapeLayer
from CellTypeSpecificLayers import DenseEILayer
from generators import StimulusDemixingTask
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

def model(input_var, batch_size=1, n_in=100, n_out=1, n_hid=200, ei_ratio=0.8):

    # Input Layer
    l_in         = InputLayer((batch_size, None, n_in), input_var=input_var)
    _, seqlen, _ = l_in.input_var.shape
    # Recurrent EI Net
    l_in_hid     = DenseEILayer(lasagne.layers.InputLayer((None, n_in)), ei_ratio, n_hid, W=lasagne.init.GlorotNormal(0.5), b=None, nonlinearity=lasagne.nonlinearities.linear, leak=1.0, diagonal=True)
    l_hid_hid    = DenseEILayer(lasagne.layers.InputLayer((None, n_hid)), ei_ratio, n_hid, W=lasagne.init.GlorotNormal(0.5), nonlinearity=lasagne.nonlinearities.linear, leak=1.0, diagonal=False)
    l_rec        = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid, nonlinearity=lasagne.nonlinearities.rectify)
    # Output Layer
    l_shp        = ReshapeLayer(l_rec, (-1, n_hid))
    l_dense      = DenseEILayer(l_shp, ei_ratio, num_units=n_out, W=lasagne.init.GlorotNormal(0.5), nonlinearity=lasagne.nonlinearities.sigmoid, leak=1.0, diagonal=True)
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out        = ReshapeLayer(l_dense, (batch_size, seqlen, n_out))

    return l_out, l_rec

if __name__ == '__main__':
    # Define the input and expected output variable
    input_var, target_var = T.tensor3s('input', 'target')
    # The generator to sample examples from
    tr_cond               = 'two_gains'
    test_cond             = 'all_gains'
    W_mix                 = np.random.rand(4, 4)
    f_I                   = np.random.rand(1, 40)
    f_b                   = np.random.rand(1, 40)
    generator             = StimulusDemixingTask(W_mix, f_I, f_b, max_iter=10001, batch_size=100, n_in=10, n_out=4, stim_dur=10, nmc = 250, tr_cond=tr_cond)
    test_generator        = StimulusDemixingTask(W_mix, f_I, f_b, max_iter=2501, batch_size=100, n_in=10, n_out=4, stim_dur=10, nmc = 250, tr_cond=test_cond)

    l_out, l_rec          = model(input_var, batch_size=generator.batch_size, n_in=generator.n_out*generator.n_in, n_out=generator.n_out, n_hid=200, ei_ratio=0.8)
    # The generated output variable and the loss function
    pred_var              = T.clip(lasagne.layers.get_output(l_out), 1e-6, 1.0 - 1e-6)
    loss                  = T.mean(lasagne.objectives.binary_crossentropy(pred_var[:,-1,:], target_var[:,-1,:])) 
    
    # Create the update expressions
    params                = lasagne.layers.get_all_params(l_out, trainable=True)
    updates               = lasagne.updates.adam(loss, params, learning_rate=0.001)
    
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
    for i, (example_input, example_output, p) in generator:
        score              = train_fn(example_input, example_output)
        example_prediction = pred_fn(example_input)
        opt_vec.append(p)
        net_vec.append(example_prediction[:,-1,:])
        if i % 500 == 0:
            opt_vec   = np.asarray(opt_vec)
            net_vec   = np.asarray(net_vec)

            inf_loss1 = np.nanmean( opt_vec[:,:,0] * np.log(opt_vec[:,:,0]/net_vec[:,:,0]) + (1.0 - opt_vec[:,:,0]) * np.log((1.0 - opt_vec[:,:,0])/(1.0 - net_vec[:,:,0])) ) / np.nanmean( opt_vec[:,:,0] * np.log(2.0*opt_vec[:,:,0]) + (1.0-opt_vec[:,:,0]) * np.log(2.0*(1.0-opt_vec[:,:,0])) ) 
            inf_loss2 = np.nanmean( opt_vec[:,:,1] * np.log(opt_vec[:,:,1]/net_vec[:,:,1]) + (1.0 - opt_vec[:,:,1]) * np.log((1.0 - opt_vec[:,:,1])/(1.0 - net_vec[:,:,1])) ) / np.nanmean( opt_vec[:,:,1] * np.log(2.0*opt_vec[:,:,1]) + (1.0-opt_vec[:,:,1]) * np.log(2.0*(1.0-opt_vec[:,:,1])) ) 
            inf_loss3 = np.nanmean( opt_vec[:,:,2] * np.log(opt_vec[:,:,2]/net_vec[:,:,2]) + (1.0 - opt_vec[:,:,2]) * np.log((1.0 - opt_vec[:,:,2])/(1.0 - net_vec[:,:,2])) ) / np.nanmean( opt_vec[:,:,2] * np.log(2.0*opt_vec[:,:,2]) + (1.0-opt_vec[:,:,2]) * np.log(2.0*(1.0-opt_vec[:,:,2])) ) 
            inf_loss4 = np.nanmean( opt_vec[:,:,3] * np.log(opt_vec[:,:,3]/net_vec[:,:,3]) + (1.0 - opt_vec[:,:,3]) * np.log((1.0 - opt_vec[:,:,3])/(1.0 - net_vec[:,:,3])) ) / np.nanmean( opt_vec[:,:,3] * np.log(2.0*opt_vec[:,:,3]) + (1.0-opt_vec[:,:,3]) * np.log(2.0*(1.0-opt_vec[:,:,3])) ) 
            inf_loss = np.nanmean([inf_loss1, inf_loss2, inf_loss3, inf_loss4])
            inf_loss_vec.append(inf_loss)
            print 'Batch #%d; Infloss: %.6f' % ( i, inf_loss )
            opt_vec   = []
            net_vec   = [] 

    # TESTING
    opt_vec, net_vec, ex_hid_vec, ex_inp_vec = [], [], [], []
    for i, (example_input, example_output, p) in test_generator:
        example_prediction = pred_fn(example_input)
        example_hidden     = rec_layer_fn(example_input)
        opt_vec.append(p)
        net_vec.append(example_prediction[:,-1,:])
        if i % 100 == 0:
            ex_hid_vec.append(example_hidden)
            ex_inp_vec.append(example_input)
        
    opt_vec   = np.asarray(opt_vec)
    net_vec   = np.asarray(net_vec)

    inf_loss1 = np.nanmean( opt_vec[:,:,0] * np.log(opt_vec[:,:,0]/net_vec[:,:,0]) + (1.0 - opt_vec[:,:,0]) * np.log((1.0 - opt_vec[:,:,0])/(1.0 - net_vec[:,:,0])) ) / np.nanmean( opt_vec[:,:,0] * np.log(2.0*opt_vec[:,:,0]) + (1.0-opt_vec[:,:,0]) * np.log(2.0*(1.0-opt_vec[:,:,0])) ) 
    inf_loss2 = np.nanmean( opt_vec[:,:,1] * np.log(opt_vec[:,:,1]/net_vec[:,:,1]) + (1.0 - opt_vec[:,:,1]) * np.log((1.0 - opt_vec[:,:,1])/(1.0 - net_vec[:,:,1])) ) / np.nanmean( opt_vec[:,:,1] * np.log(2.0*opt_vec[:,:,1]) + (1.0-opt_vec[:,:,1]) * np.log(2.0*(1.0-opt_vec[:,:,1])) ) 
    inf_loss3 = np.nanmean( opt_vec[:,:,2] * np.log(opt_vec[:,:,2]/net_vec[:,:,2]) + (1.0 - opt_vec[:,:,2]) * np.log((1.0 - opt_vec[:,:,2])/(1.0 - net_vec[:,:,2])) ) / np.nanmean( opt_vec[:,:,2] * np.log(2.0*opt_vec[:,:,2]) + (1.0-opt_vec[:,:,2]) * np.log(2.0*(1.0-opt_vec[:,:,2])) ) 
    inf_loss4 = np.nanmean( opt_vec[:,:,3] * np.log(opt_vec[:,:,3]/net_vec[:,:,3]) + (1.0 - opt_vec[:,:,3]) * np.log((1.0 - opt_vec[:,:,3])/(1.0 - net_vec[:,:,3])) ) / np.nanmean( opt_vec[:,:,3] * np.log(2.0*opt_vec[:,:,3]) + (1.0-opt_vec[:,:,3]) * np.log(2.0*(1.0-opt_vec[:,:,3])) ) 
 
    inf_loss_test = np.nanmean([inf_loss1, inf_loss2, inf_loss3, inf_loss4])
    print 'Test data; Infloss: %.6f'%inf_loss_test
                
    ex_hid_vec = np.asarray(ex_hid_vec)    
    ex_hid_vec = np.reshape(ex_hid_vec,(-1, test_generator.stim_dur, 200))    

    ex_inp_vec = np.asarray(ex_inp_vec)    
    ex_inp_vec = np.reshape(ex_inp_vec,(-1, test_generator.stim_dur, 40))    

    # Prepare the actual connectivity matrices
    D_in  = np.diag( np.concatenate( (np.ones(32),-np.ones(8)) ) )         
    W_in  = lasagne.layers.get_all_param_values(l_out, trainable=True)[0]
    W_in  = np.maximum( W_in, 0.0 )
    W_in  = np.dot( D_in, W_in )

    D_hid = np.diag( np.concatenate( (np.ones(160),-np.ones(40)) ) )
    W_hid = lasagne.layers.get_all_param_values(l_out, trainable=True)[1]
    W_hid = np.maximum( W_hid, 0.0 )
    W_hid[np.diag_indices(200)] = 0.0
    W_hid = np.dot(D_hid, W_hid)

    D_out = np.diag( np.concatenate( (np.ones(160),-np.ones(40)) ) )
    w_out = lasagne.layers.get_all_param_values(l_out, trainable=True)[3]
    w_out = np.maximum( w_out, 0.0 )
    w_out = np.dot( D_out, w_out )
    
    # SAVE TRAINED MODEL  
    sio.savemat('ei_sd_testinfloss_twogains_run%i.mat'%job_idx, {'inf_loss_test':inf_loss_test, 'opt_vec':opt_vec, 'net_vec':net_vec})      
#    sio.savemat('ei_sd_everything_allgains_run%i.mat'%job_idx, {'W_in':  W_in, 
#                                                                'W_hid': W_hid,
#                                                                'b_hid': lasagne.layers.get_all_param_values(l_out, trainable=True)[2],
#                                                                'w_out': w_out,
#                                                                'b_out': lasagne.layers.get_all_param_values(l_out, trainable=True)[4],
#                                                                'inf_loss_vec':np.asarray(inf_loss_vec), 
#                                                                'inpResps':ex_inp_vec, 
#                                                                'hidResps':ex_hid_vec} )           