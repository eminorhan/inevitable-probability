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
from generators import StimulusDemixingTaskFFWD
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

nnn              = np.ceil(np.logspace(.3,2.2,14))
nhu_vec, nin_vec = np.meshgrid(nnn, nnn) 
nhu_vec          = nhu_vec.flatten()
nin_vec          = nin_vec.flatten()
n_in             = int(nin_vec[job_idx-1])
n_hid            = int(nhu_vec[job_idx-1])

def model(input_var, batch_size=1, n_in=100, n_out=1, n_hid=200):

    # Input Layer
    l_in         = InputLayer((batch_size, n_in), input_var=input_var)
    # Recurrent EI Net
    l_in_hid     = DenseLayer(l_in, n_hid, nonlinearity=lasagne.nonlinearities.rectify)

    # Output Layer
    l_shp        = ReshapeLayer(l_in_hid, (-1, n_hid))
    l_dense      = DenseLayer(l_shp, num_units=n_out, nonlinearity=lasagne.nonlinearities.sigmoid)
    # To reshape back to our original shape, we can use the symbolic shape variables we retrieved above.
    l_out        = ReshapeLayer(l_dense, (batch_size, n_out))

    return l_out, l_in_hid

if __name__ == '__main__':
    # Define the input and expected output variable
    input_var, target_var = T.fmatrices('input', 'target')
    # The generator to sample examples from
    tr_cond               = 'all_gains'
    test_cond             = 'all_gains'
    W_mix                 = np.random.rand(4, 4)
    f_I                   = np.random.rand(1, 4*n_in)
    f_b                   = np.random.rand(1, 4*n_in)
    generator             = StimulusDemixingTaskFFWD(W_mix, f_I, f_b, max_iter=100001, batch_size=100, n_in=n_in, n_out=4, nmc = 250, tr_cond=tr_cond)
    l_out, l_rec          = model(input_var, batch_size=generator.batch_size, n_in=generator.n_out*generator.n_in, n_out=generator.n_out, n_hid=n_hid)

    # The generated output variable and the loss function
#    all_layers            = lasagne.layers.get_all_layers(l_out)
#    l2_penalty            = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 1e-6
    pred_var              = T.clip(lasagne.layers.get_output(l_out), 1e-6, 1.0 - 1e-6)
    loss                  = T.mean(lasagne.objectives.binary_crossentropy(pred_var, target_var)) # + l2_penalty
    
    # Create the update expressions
    params                = lasagne.layers.get_all_params(l_out, trainable=True)
    updates               = lasagne.updates.adam(loss, params, learning_rate=0.0005) 
    
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
    success = 0.0
    opt_vec, net_vec, inf_loss_vec = [], [], []
    for i, (example_input, example_output, p) in generator:
        score              = train_fn(example_input, example_output)
        example_prediction = pred_fn(example_input)
        opt_vec.append(p)
        net_vec.append(example_prediction)
        if i % 500 == 0:
            opt_vec   = np.asarray(opt_vec)
            net_vec   = np.asarray(net_vec)
            inf_loss1 = np.nanmean( opt_vec[:,:,0] * np.log(opt_vec[:,:,0]/net_vec[:,:,0]) + (1.0 - opt_vec[:,:,0]) * np.log((1.0 - opt_vec[:,:,0])/(1.0 - net_vec[:,:,0])) ) / np.nanmean( opt_vec[:,:,0] * np.log(2.0*opt_vec[:,:,0]) + (1.0-opt_vec[:,:,0]) * np.log(2.0*(1.0-opt_vec[:,:,0])) ) 
            inf_loss2 = np.nanmean( opt_vec[:,:,1] * np.log(opt_vec[:,:,1]/net_vec[:,:,1]) + (1.0 - opt_vec[:,:,1]) * np.log((1.0 - opt_vec[:,:,1])/(1.0 - net_vec[:,:,1])) ) / np.nanmean( opt_vec[:,:,1] * np.log(2.0*opt_vec[:,:,1]) + (1.0-opt_vec[:,:,1]) * np.log(2.0*(1.0-opt_vec[:,:,1])) ) 
            inf_loss3 = np.nanmean( opt_vec[:,:,2] * np.log(opt_vec[:,:,2]/net_vec[:,:,2]) + (1.0 - opt_vec[:,:,2]) * np.log((1.0 - opt_vec[:,:,2])/(1.0 - net_vec[:,:,2])) ) / np.nanmean( opt_vec[:,:,2] * np.log(2.0*opt_vec[:,:,2]) + (1.0-opt_vec[:,:,2]) * np.log(2.0*(1.0-opt_vec[:,:,2])) ) 
            inf_loss4 = np.nanmean( opt_vec[:,:,3] * np.log(opt_vec[:,:,3]/net_vec[:,:,3]) + (1.0 - opt_vec[:,:,3]) * np.log((1.0 - opt_vec[:,:,3])/(1.0 - net_vec[:,:,3])) ) / np.nanmean( opt_vec[:,:,3] * np.log(2.0*opt_vec[:,:,3]) + (1.0-opt_vec[:,:,3]) * np.log(2.0*(1.0-opt_vec[:,:,3])) ) 
            inf_loss  = np.nanmean([inf_loss1, inf_loss2, inf_loss3, inf_loss4])
            inf_loss_vec.append(inf_loss)            
            print 'Batch #%d; Infloss: %.6f' % (i, inf_loss )
            if inf_loss < 0.1:
                success = 1.0
                break;            
            opt_vec   = []
            net_vec   = [] 
            
    # SAVE TRAINED MODEL  
    sio.savemat('sd_nin%i_nhu%i_jobidx%i.mat'%(n_in,n_hid,job_idx), {'inf_loss':inf_loss, 'inf_loss_vec':np.asarray(inf_loss_vec), 'success':success } )      