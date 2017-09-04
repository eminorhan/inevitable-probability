# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:53:50 2016 by emin
"""
import theano
import numpy as np

class Task(object):

    def __init__(self, max_iter=None, batch_size=1):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1) , self.sample()
        else:
            raise StopIteration()

    def sample(self):
        raise NotImplementedError()
        

class BinaryCategorizationTask(Task):
    ''' Parameters  '''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigma_sq=100.0, tr_cond='all_gains'):
        super(BinaryCategorizationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.stim_dur  = stim_dur
        self.sigma_sq  = sigma_sq
        self.tr_cond   = tr_cond
        self.phi       = np.concatenate((np.linspace(-40.0, 40.0, 0.8*self.n_in),np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in))))
        self.sigma_sq1 = 9.0
        self.sigma_sq2 = 144.0

    def sample(self):

        c              = np.random.choice([0.0, 1.0], size=(self.batch_size, 1))
        c1ind          = np.where(c==1.0)[0]        

        s              = 12.0 * np.random.randn(1, self.batch_size)
        s[0, c1ind]    = 3.0 * np.random.randn(1, c1ind.size)
        s              = np.repeat(s,self.n_in,axis=0).T
        s              = np.tile(s,(self.stim_dur,1,1))
        s              = np.swapaxes(s,0,1)

        if self.tr_cond == 'all_gains':
            g = (0.4 / self.stim_dur) * np.random.choice([0.37, 0.90, 1.81, 2.82, 3.57, 4.00],size=(1,self.batch_size))
            g = np.repeat(g,self.n_in,axis=0).T
            g = np.tile(g,(self.stim_dur,1,1))
            g = np.swapaxes(g,0,1)
        else:
            g = (0.4 / self.stim_dur) * 4.20 * np.ones((1,self.batch_size))
            g = np.repeat(g,self.n_in,axis=0).T
            g = np.tile(g,(self.stim_dur,1,1))
            g = np.swapaxes(g,0,1)

        rate = g * np.exp(- ( (np.tile(self.phi, (self.batch_size, self.stim_dur, 1) ) - s ) / (np.sqrt(2.0 * self.sigma_sq))) ** 2)   
        resp = np.random.poisson(rate, size=(self.batch_size, self.stim_dur, self.n_in))
        
        example_input         = resp
        example_output        = c
        
        r                     = np.squeeze(np.sum(resp,axis=1))
        a                     = np.ones(shape=(self.n_in, )) / self.sigma_sq
        e                     = self.phi / self.sigma_sq
        aux_2                 = 1.0 + self.sigma_sq2 * np.dot(r,a)
        aux_1                 = 1.0 + self.sigma_sq1 * np.dot(r,a)
        aux_3                 = (self.sigma_sq2 - self.sigma_sq1) * (np.dot(r,e)) ** 2
        d                     = 0.5 * (np.log( aux_2 / aux_1 ) - (aux_3 / (aux_1 * aux_2)) )
        p                     = 1.0 / (1.0 + np.exp(-d))

        return example_input, example_output, g, c, p


class BinaryCategorizationTaskFFWD(Task):
    ''' Parameters  '''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, sigma_sq=100.0, tr_cond='all_gains'):
        super(BinaryCategorizationTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.sigma_sq  = sigma_sq
        self.tr_cond   = tr_cond
        self.phi       = np.linspace(-40.0, 40.0,self.n_in)
        self.sigma_sq1 = 9.0
        self.sigma_sq2 = 144.0

    def sample(self):
        
        c              = np.random.choice([0.0, 1.0], size=(self.batch_size, 1))
        c1ind          = np.where(c==1.0)[0]        

        s              = 12.0 * np.random.randn(1, self.batch_size)
        s[0, c1ind]    = 3.0  * np.random.randn(1,c1ind.size)
        s              = np.repeat(s,self.n_in,axis=0).T

        if self.tr_cond == 'all_gains':
            g         = 0.4 * np.random.choice([0.37, 0.90, 1.81, 2.82, 3.57, 4.00], size=(1,self.batch_size))
            g         = np.repeat(g,self.n_in,axis=0).T
        else:
            g         = 0.4 * 4.20 * np.ones((1,self.batch_size))
            g         = np.repeat(g,self.n_in,axis=0).T


        rate     = g * np.exp( - (s - np.tile(self.phi, (self.batch_size, 1) ) )**2 / (2.0 * self.sigma_sq) )
        resp     = np.random.poisson(rate, size=(self.batch_size, self.n_in))
        
        example_input         = resp
        example_output        = np.zeros((self.batch_size, self.n_out), dtype=theano.config.floatX)
        example_output[:,:]   = c
        r                     = np.squeeze(resp)
        a                     = np.ones(shape=(self.n_in, )) / self.sigma_sq
        e                     = self.phi / self.sigma_sq
        aux_2                 = 1.0 + self.sigma_sq2 * np.dot(r,a)
        aux_1                 = 1.0 + self.sigma_sq1 * np.dot(r,a)
        aux_3                 = (self.sigma_sq2 - self.sigma_sq1) * (np.dot(r,e)) ** 2
        d                     = 0.5 * (np.log( aux_2 / aux_1 ) - (aux_3 / (aux_1 * aux_2)) )
        p                     = 1.0 / (1.0 + np.exp(-d))

        return example_input, example_output, g, c, p


class CausalInferenceTask(Task):
    ''' Parameters '''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigma_sq=100.0, tr_cond='all_gains'):
        super(CausalInferenceTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.stim_dur  = stim_dur
        self.sigma_sq  = sigma_sq
        self.tr_cond   = tr_cond
        self.phi       = np.concatenate( ( np.linspace(-40.0, 40.0, 0.8*self.n_in), np.linspace(-40.0, 40.0, np.ceil(0.8*self.n_in)), np.linspace(-40.0, 40.0, 0.2*self.n_in), np.linspace(-40.0, 40.0, np.ceil(0.2*self.n_in)) ) )
        self.JS        = 1.0 / 100.0

    def sample(self):

        # Same cause or different causes 
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size, 1))
        C1ind          = np.where(C==1.0)[0]        

        S              = 10.0 * np.random.randn(2, self.batch_size)
        S[1, C1ind]    = S[0, C1ind] 
        S              = np.repeat(S,self.n_in,axis=0).T
        S              = np.tile(S,(self.stim_dur,1,1))
        S              = np.swapaxes(S,0,1)
        
        s1e = S[:,:,:0.8*self.n_in]
        s1i = S[:,:,0.8*self.n_in:self.n_in]
        s2e = S[:,:,self.n_in:1.8*self.n_in]
        s2i = S[:,:,1.8*self.n_in:]    
        
        S = np.concatenate((s1e,s2e,s1i,s2i),axis=2)

        # Gains
        if self.tr_cond == 'all_gains':
            G1 = (1.0/self.stim_dur) * np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:self.n_in*0.2]

            G2 = (1.0/self.stim_dur) * np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5], size=(1,self.batch_size))
            G2 = np.repeat(G2,self.n_in*0.8,axis=0).T
            G2 = np.tile(G2,(self.stim_dur,1,1))
            G2 = np.swapaxes(G2,0,1)
            G4 = G2[:,:,:self.n_in*0.2]

            G  = np.concatenate( (G1,G2,G3,G4), axis=2 ) 
            
        else:
            G1 = (1.0/self.stim_dur) * np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:self.n_in*0.2]

            G2 = G1
            G4 = G2[:,:,:self.n_in*0.2]

            G  = np.concatenate( (G1,G2,G3,G4), axis=2 ) 

        # Noisy responses
        Lambda         = G * np.exp( - (S - np.tile(self.phi, (self.batch_size, self.stim_dur, 1) ) )**2 / (2.0 * self.sigma_sq) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = np.repeat(C, self.stim_dur, axis=1)
        example_output = np.expand_dims(example_output,axis=2)

        R1e = R[:,:,:0.8*self.n_in]
        R2e = R[:,:,0.8*self.n_in:1.6*self.n_in]
        R1i = R[:,:,1.6*self.n_in:1.8*self.n_in]
        R2i = R[:,:,1.8*self.n_in:]    
        
        R1    = np.concatenate((R1e,R1i),axis=2)
        R2    = np.concatenate((R2e,R2i),axis=2)

        r1    = np.squeeze(np.sum(R1,axis=1))
        r2    = np.squeeze(np.sum(R2,axis=1))

        a     = np.ones(shape=(self.n_in,)) / self.sigma_sq
        e     = np.concatenate( ( np.linspace(-40.0, 40.0, 0.8*self.n_in), np.linspace(-40.0, 40.0, np.ceil(0.2*self.n_in)) )   ) / self.sigma_sq
        
        z11   = np.dot(r1,e)
        z12   = np.dot(r1,a)
        z21   = np.dot(r2,e)
        z22   = np.dot(r2,a)
        
        aux_1 = (z11 * z21) / (z12 + z22 + self.JS) 
        aux_2 = (z22 * z11**2) / ((z12 + self.JS)*(z12 + z22 + self.JS))
        aux_3 = (z12 * z21**2) / ((z22 + self.JS)*(z12 + z22 + self.JS))
        aux_4 = (z12 * z22) / (self.JS * (z12 + z22 + self.JS))

        d     = aux_1 - 0.5 * (aux_2 + aux_3 - np.log(1.0 + aux_4))
        p     = 1.0 / (1.0 + np.exp(-d))

        return example_input, example_output, p


class CausalInferenceTaskFFWD(Task):
    ''' Parameters '''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, sigma_sq=100.0, tr_cond='all_gains'):
        super(CausalInferenceTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.sigma_sq  = sigma_sq
        self.tr_cond   = tr_cond
        self.phi       = np.linspace(-40.0, 40.0, self.n_in)
        self.JS        = 1.0 / 100.0

    def sample(self):

        # Same cause or different causes 
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size, 1))
        C1ind          = np.where(C==1.0)[0]        

        S              = 10.0 * np.random.randn(2, self.batch_size)
        S[1, C1ind]    = S[0, C1ind] 
        S              = np.repeat(S,self.n_in,axis=0).T

        # Gains
        if self.tr_cond == 'all_gains':
            G = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5], size=(2,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
        else:
            G = np.random.choice([0.5, 2.5], size=(1,self.batch_size))
            G = np.repeat(G, 2 * self.n_in, axis=0).T

        # Noisy responses
        Lambda         = G * np.exp( - (S - np.tile(self.phi, (self.batch_size, 2) ) )**2 / (2.0 * self.sigma_sq) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = C

        r1 = R[:,0:self.n_in]
        r2 = R[:,self.n_in:]

        a = np.ones(shape=(self.n_in,)) / self.sigma_sq
        e = self.phi / self.sigma_sq
        
        z11 = np.dot(r1,e)
        z12 = np.dot(r1,a)
        z21 = np.dot(r2,e)
        z22 = np.dot(r2,a)
        
        aux_1 = (z11 * z21) / (z12 + z22 + self.JS) 
        aux_2 = (z22 * z11**2) / ((z12 + self.JS)*(z12 + z22 + self.JS))
        aux_3 = (z12 * z21**2) / ((z22 + self.JS)*(z12 + z22 + self.JS))
        aux_4 = (z12 * z22) / (self.JS * (z12 + z22 + self.JS))

        d = aux_1 - 0.5 * (aux_2 + aux_3 - np.log(1.0 + aux_4))
        p = 1.0 / (1.0 + np.exp(-d))

        return example_input, example_output, p
        
        
class CueCombinationTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigma_sq=100.0, tr_cond='all_gains'):
        super(CueCombinationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.stim_dur   = stim_dur
        self.sigma_sq   = sigma_sq
        self.tr_cond    = tr_cond
        # Exc1-Exc2-Inh1-Inh2
        self.phis       = np.concatenate( ( np.linspace(-40.0, 40.0, 0.8*self.n_in), np.linspace(-40.0, 40.0, 0.8*self.n_in), np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in)), np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in)) ) )
        self.ngp        = 100
        self.ss         = np.linspace( -20.0, 20.0, self.ngp )

    def sample(self):
        
        s = 40.0 * np.random.rand(1, self.batch_size) - 20.0   
        s = np.repeat(s,2 * self.n_in,axis=0).T
        s = np.tile(s,(self.stim_dur,1,1))
        s = np.swapaxes(s,0,1)

        if self.tr_cond == 'all_gains':
            G1 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:self.n_in*0.2]

            G2 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G2 = np.repeat(G2,self.n_in*0.8,axis=0).T
            G2 = np.tile(G2,(self.stim_dur,1,1))
            G2 = np.swapaxes(G2,0,1)
            G4 = G2[:,:,:self.n_in*0.2]

            G  = np.concatenate( (G1,G2,G3,G4), axis=2 ) 
            
        else:
            G1 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:self.n_in*0.2]

            G2 = G1
            G4 = G2[:,:,:self.n_in*0.2]

            G  = np.concatenate( (G1,G2,G3,G4), axis=2 ) 

        # Noisy responses
        Lambda         = G * np.exp( - (s - np.tile(self.phis, (self.batch_size, self.stim_dur, 1) ) )**2 / (2.0 * self.sigma_sq) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = s

        opt_s          = np.zeros((self.batch_size,))
        
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,))
            for j in range(2*self.n_in):
                pr = pr - np.sum(R[i,:,j]) * (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) - np.sum(G[i,:,j]) * np.exp( - (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            pr = np.exp(pr)
            pr = pr / np.sum(pr)
            opt_s[i] = np.dot(pr,self.ss)
            
        return example_input, example_output, G[:,0,0], G[:,0,-1], s[:,0,0], opt_s        


class ModularCueCombinationTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigma_sq=100.0, tr_cond='all_gains'):
        super(ModularCueCombinationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.stim_dur   = stim_dur
        self.sigma_sq   = sigma_sq
        self.tr_cond    = tr_cond
        # Exc1-Exc2-Inh1-Inh2
        self.phis       = np.concatenate( ( np.linspace(-40.0, 40.0, int(0.8*self.n_in)), 
                                            np.linspace(-40.0, 40.0, int(0.8*self.n_in)), 
                                            np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in)), 
                                            np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in)),
                                            np.linspace(-40.0, 40.0, int(0.8*self.n_in)), 
                                            np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in))  ) )
        self.ngp        = 100
        self.ss         = np.linspace( -20.0, 20.0, self.ngp )

    def sample(self):
        
        s = 40.0 * np.random.rand(1, self.batch_size) - 20.0   
        s = np.repeat(s,3 * self.n_in,axis=0).T
        s = np.tile(s,(self.stim_dur,1,1))
        s = np.swapaxes(s,0,1)

        if self.tr_cond == 'all_gains':
            G1 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:int(self.n_in*0.2)]

            G2 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G2 = np.repeat(G2,self.n_in*0.8,axis=0).T
            G2 = np.tile(G2,(self.stim_dur,1,1))
            G2 = np.swapaxes(G2,0,1)
            G4 = G2[:,:,:int(self.n_in*0.2)]

            G5 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G5 = np.repeat(G5,self.n_in*0.8,axis=0).T
            G5 = np.tile(G5,(self.stim_dur,1,1))
            G5 = np.swapaxes(G5,0,1)
            G6 = G5[:,:,:int(self.n_in*0.2)]

            G  = np.concatenate( (G1,G2,G3,G4,G5,G6), axis=2 ) 
            
        else:
            G1 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:int(self.n_in*0.2)]

            G2 = G1
            G4 = G2[:,:,:int(self.n_in*0.2)]

            G5 = G1
            G6 = G5[:,:,:int(self.n_in*0.2)]

            G  = np.concatenate( (G1,G2,G3,G4,G5,G6), axis=2 ) 

        # Noisy responses
        Lambda         = G * np.exp( - (s - np.tile(self.phis, (self.batch_size, self.stim_dur, 1) ) )**2 / (2.0 * self.sigma_sq) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = s

        opt_s          = np.zeros((self.batch_size,))
        
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,))
            for j in range(3*self.n_in):
                pr = pr - np.sum(R[i,:,j]) * (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) - np.sum(G[i,:,j]) * np.exp( - (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            pr = np.exp(pr)
            pr = pr / np.sum(pr)
            opt_s[i] = np.dot(pr,self.ss)
            
        return example_input, example_output, G[:,0,0], G[:,0,-1], s[:,0,0], opt_s        



class CueCombinationTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, sigma_sq=100.0, tr_cond='all_gains'):
        super(CueCombinationTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.sigma_sq   = sigma_sq
        self.tr_cond    = tr_cond
        self.phi        = np.linspace(-40.0, 40.0, self.n_in)
        self.phis       = np.tile(self.phi, 2 )
        self.ngp        = 2500
        self.ss         = np.linspace(-20.0,20.0,self.ngp)

    def sample(self):
        
        S = 40.0 * np.random.rand(1, self.batch_size) - 20.0   
        S = np.repeat(S,2 * self.n_in,axis=0).T
       
        
        if self.tr_cond == 'all_gains':
            G = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25], size=(2,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
        else:
            G = np.random.choice([0.25,1.25], size=(1,self.batch_size))
            G = np.repeat(G, 2 * self.n_in, axis=0).T

        Lambda = G * np.exp( - (S - np.tile(self.phi, (self.batch_size, 2) ) )**2 / (2.0 * self.sigma_sq) )
        R      = np.random.poisson(Lambda)
        
        
        example_input  = R
        example_output = np.expand_dims(S[:,0], axis=1)

        resp_1         = R[:,0:self.n_in]
        resp_2         = R[:,self.n_in:]
        
        example_input  = np.concatenate((resp_1,resp_2), axis=1)
        opt_s          = np.zeros((self.batch_size,))
        
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,))
            for j in range(2*self.n_in):
                pr = pr - R[i,j] * (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) - G[i,j] * np.exp( - (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            pr = np.exp(pr)
            pr = pr / np.sum(pr)
            opt_s[i] = np.dot(pr,self.ss) # posterior mean
            
            # compute posterior median
#            cpr = np.cumsum(pr)
#            indx = np.where(cpr>0.5)[0][0]
#            opt_s[i] = 0.5*(self.ss[indx]+self.ss[indx-1])
            
        return example_input, example_output, G[:,0], G[:,-1], S[:,0], opt_s        


class ModularCueCombinationTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, sigma_sq=100.0, tr_cond='all_gains'):
        super(ModularCueCombinationTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.sigma_sq   = sigma_sq
        self.tr_cond    = tr_cond
        self.phi        = np.linspace(-40.0, 40.0, self.n_in)
        self.phis       = np.tile(self.phi, 3 )
        self.ngp        = 2500
        self.ss         = np.linspace(-20.0,20.0,self.ngp)

    def sample(self):
        
        S = 40.0 * np.random.rand(1, self.batch_size) - 20.0   
        S = np.repeat(S, 3 * self.n_in,axis=0).T
       
        
        if self.tr_cond == 'all_gains':
            G = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25], size=(3,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
        else:
            G = np.random.choice([0.25,1.25], size=(1,self.batch_size))
            G = np.repeat(G, 3 * self.n_in, axis=0).T

        Lambda = G * np.exp( - (S - np.tile(self.phi, (self.batch_size, 3) ) )**2 / (2.0 * self.sigma_sq) )
        R      = np.random.poisson(Lambda)
        
        
        example_input  = R
        example_output = np.expand_dims(S[:,0], axis=1)

        resp_1         = R[:,0:self.n_in]
        resp_2         = R[:,self.n_in:(2*self.n_in)]
        resp_3         = R[:,(2*self.n_in):]
        
        example_input  = np.concatenate((resp_1,resp_2,resp_3), axis=1)
        opt_s          = np.zeros((self.batch_size,))
        
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,))
            for j in range(3*self.n_in):
                pr = pr - R[i,j] * (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) - G[i,j] * np.exp( - (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            pr = np.exp(pr)
            pr = pr / np.sum(pr)
            opt_s[i] = np.dot(pr,self.ss) # posterior mean 

            # compute posterior median
#            cpr = np.cumsum(pr)
#            indx = np.where(cpr>0.5)[0][0]
#            opt_s[i] = 0.5*(self.ss[indx]+self.ss[indx-1])
                        
        return example_input, example_output, G[:,0], G[:,-1], S[:,0], opt_s        


class CueCombinationFetschTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, sigma_sq=100.0, alpha=2.0, beta=0.6, tr_cond='all_gains'):
        super(CueCombinationFetschTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.sigma_sq   = sigma_sq
        self.alpha      = alpha
        self.beta       = beta
        self.tr_cond    = tr_cond
        self.phi        = np.linspace(-40.0, 40.0, self.n_in)
        self.phis       = np.tile(self.phi, 2 )
        self.ngp        = 100
        self.ss         = np.linspace(-20.0,20.0,self.ngp)

    def sample(self):
        
        S = 40.0 * np.random.rand(1, self.batch_size) - 20.0   
        S = np.repeat(S,2 * self.n_in,axis=0).T  
        
        if self.tr_cond == 'all_gains':
            G = np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=(2,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
        else:
            G = np.random.choice([0.2, 0.8], size=(1,self.batch_size))
            G = np.repeat(G, 2 * self.n_in, axis=0).T

        Lambda = G * self.alpha * np.exp( - (S - np.tile(self.phi, (self.batch_size, 2) ) )**2 / (2.0 * self.sigma_sq) ) + (1.0-G) * self.beta
        R      = np.random.poisson(Lambda)
        
        example_input  = R
        example_output = np.expand_dims(S[:,0], axis=1)

        resp_1         = R[:,0:self.n_in]
        resp_2         = R[:,self.n_in:]
        
        example_input  = np.concatenate((resp_1,resp_2), axis=1)
        opt_s = np.zeros((self.batch_size,))
        
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,))
            for j in range(2*self.n_in):
                aux = G[i,j] * self.alpha * np.exp( - (self.ss - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) + (1.0-G[i,j]) * self.beta
                pr  = pr + R[i,j] * np.log( aux ) - aux
            pr = np.exp(pr)
            pr = pr / np.sum(pr)
            opt_s[i] = np.dot(pr,self.ss)
                        
        return example_input, example_output, G[:,0], G[:,-1], S[:,0], opt_s        


class CueCombinationSpeedTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, alpha=10.0, beta=2.0, gamma=3.0, r_0=0.5, A=5.0, s_0=1.0, sigtc_sq=1.0, tr_cond='all_gains'):
        super(CueCombinationSpeedTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.sigtc_sq   = sigtc_sq
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.tr_cond    = tr_cond
        self.phi        = np.logspace(-1.0, 2.0, self.n_in)
        self.phis       = np.tile(self.phi, 2 )
        self.ngp        = 100
        self.ss         = np.linspace(1.0,64.0,self.ngp)
        self.r_0        = r_0
        self.A          = A
        self.s_0        = s_0

    def sample(self):
        
        S = 63.0 * np.random.rand(1, self.batch_size) + 1.0   
        S = np.repeat(S,2 * self.n_in,axis=0).T
       
        
        if self.tr_cond == 'all_gains':
            G = np.random.choice([0.05, 0.1, 0.2, 0.4, 1.0], size=(2,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = 1.0 / ( (self.alpha * G)**(-self.beta) +self.gamma )
        else:
            G = np.random.choice([0.1, 0.4], size=(1,self.batch_size))
            G = np.repeat(G, 2 * self.n_in, axis=0).T
            G = 1.0 / ( (self.alpha * G)**(-self.beta) +self.gamma )

        Lambda = self.r_0 + self.A * G * np.exp( -0.5*(1.0/self.sigtc_sq) * (np.log( (S+self.s_0) / (10.0*G*np.tile(self.phi, (self.batch_size, 2) ) + self.s_0) ) )**2)
        R      = np.random.poisson(Lambda)
        
        example_input  = R
        example_output = np.expand_dims(S[:,0], axis=1)

        resp_1         = R[:,0:self.n_in]
        resp_2         = R[:,self.n_in:]
        
        example_input  = np.concatenate((resp_1,resp_2), axis=1)
        opt_s = np.zeros((self.batch_size,))
        
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,))
            for j in range(2*self.n_in):
                aux = self.r_0 + self.A * G[i,j] * np.exp( -0.5*(1.0/self.sigtc_sq) * (np.log( (self.ss+self.s_0) / (10.0*G[i,j]*self.phis[j] + self.s_0) ) )**2)
                pr  = pr + R[i,j] * np.log( aux ) - aux
            pr = np.exp(pr)
            pr = pr / np.sum(pr)
            opt_s[i] = np.dot(pr,self.ss)
                        
        return example_input, example_output, G[:,0], G[:,-1], S[:,0], opt_s        

    
class CoordinateTransformationTask(Task):
    ''' Parameters '''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigma_sq=100.0, tr_cond='all_gains'):
        super(CoordinateTransformationTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.stim_dur  = stim_dur
        self.sigma_sq  = sigma_sq
        self.tr_cond   = tr_cond
        # Exc1-Exc2-Inh1-Inh2
        self.phis      = np.concatenate( ( np.linspace(-40.0, 40.0, 0.8*self.n_in), np.linspace(-40.0, 40.0, 0.8*self.n_in), np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in)), np.linspace(-40.0, 40.0, np.ceil((1.0-0.8)*self.n_in)) ) )
        self.ngp       = 50
        self.ss1, self.ss2 = np.meshgrid(np.linspace(-20.0,20.0,self.ngp),np.linspace(-20.0,20.0,self.ngp))
        self.ss        = self.ss1 + self.ss2 

    def sample(self):
        
        s  = 40.0 * np.random.rand(2, self.batch_size) - 20.0 
        st = np.sum(s, axis=0, keepdims=True) 
        st = np.repeat(st,2*self.n_in,axis=0).T
        st = np.tile(st,(self.stim_dur,1,1))
        st = np.swapaxes(st,0,1)

        s1 = np.expand_dims(s[0,:], axis=0)
        s2 = np.expand_dims(s[1,:], axis=0)
        
        s1e  = np.repeat(s1,0.8*self.n_in,axis=0).T
        s1i  = np.repeat(s1,0.2*self.n_in,axis=0).T
        s2e  = np.repeat(s2,0.8*self.n_in,axis=0).T
        s2i  = np.repeat(s2,0.2*self.n_in,axis=0).T     
        
        s  = np.concatenate((s1e,s2e,s1i,s2i),axis=1)
        s  = np.tile(s,(self.stim_dur,1,1))
        s  = np.swapaxes(s,0,1)

        if self.tr_cond == 'all_gains':
            G1 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:self.n_in*0.2]

            G2 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G2 = np.repeat(G2,self.n_in*0.8,axis=0).T
            G2 = np.tile(G2,(self.stim_dur,1,1))
            G2 = np.swapaxes(G2,0,1)
            G4 = G2[:,:,:self.n_in*0.2]

            G  = np.concatenate( (G1,G2,G3,G4), axis=2 ) 
            
        else:
            G1 = (1.0/self.stim_dur) * np.random.choice([0.25, 0.50, 0.75, 1.0, 1.25], size=(1,self.batch_size))
            G1 = np.repeat(G1,self.n_in*0.8,axis=0).T
            G1 = np.tile(G1,(self.stim_dur,1,1))
            G1 = np.swapaxes(G1,0,1)
            G3 = G1[:,:,:self.n_in*0.2]

            G2 = G1
            G4 = G2[:,:,:self.n_in*0.2]

            G  = np.concatenate( (G1,G2,G3,G4), axis=2 ) 

        # Noisy responses
        Lambda         = G * np.exp( - (s - np.tile(self.phis, (self.batch_size, self.stim_dur, 1) ) )**2 / (2.0 * self.sigma_sq) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = st

        opt_s          = np.zeros((self.batch_size,))
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,self.ngp))
            for j in range( int(0.8*self.n_in) ):
                pr = pr - np.sum(R[i,:,j]) * (self.ss1 - self.phis[j] )**2 / (2.0 * self.sigma_sq) - np.sum(G[i,:,j]) * np.exp( - (self.ss1 - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            for j in range( int(0.8*self.n_in), int(1.6*self.n_in) ):
                pr = pr - np.sum(R[i,:,j]) * (self.ss2 - self.phis[j] )**2 / (2.0 * self.sigma_sq) - np.sum(G[i,:,j]) * np.exp( - (self.ss2 - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            for j in range( int(1.6*self.n_in), int(1.8*self.n_in)):
                pr = pr - np.sum(R[i,:,j]) * (self.ss1 - self.phis[j] )**2 / (2.0 * self.sigma_sq) - np.sum(G[i,:,j]) * np.exp( - (self.ss1 - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            for j in range( int(1.8*self.n_in), int(2*self.n_in)):
                pr = pr - np.sum(R[i,:,j]) * (self.ss2 - self.phis[j] )**2 / (2.0 * self.sigma_sq) - np.sum(G[i,:,j]) * np.exp( - (self.ss2 - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            
            pr       = np.exp(pr)
            pr       = pr / np.sum(pr)
            opt_s[i] = np.dot(pr.flatten(),self.ss.flatten())

        return example_input, example_output, G[:,0,0], G[:,0,-1], st[:,0,0], opt_s


class CoordinateTransformationTaskFFWD(Task):
    ''' Parameters '''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, sigma_sq=100.0, tr_cond='all_gains'):
        super(CoordinateTransformationTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.sigma_sq  = sigma_sq
        self.tr_cond   = tr_cond
        self.phi       = np.linspace(-40.0, 40.0, self.n_in)
        self.phis      = np.tile(self.phi, 2 )
        self.ngp       = 50
        self.ss1, self.ss2 = np.meshgrid(np.linspace(-20.0,20.0,self.ngp),np.linspace(-20,20,self.ngp))
        self.ss        = self.ss1 + self.ss2 

    def sample(self):
        
        S  = 40.0 * np.random.rand(2, self.batch_size) - 20.0 
        ST = np.sum(S, axis=0, keepdims=True) 
        S  = np.repeat(S,self.n_in,axis=0).T
        
        if self.tr_cond == 'all_gains':
            G = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25], size=(2,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
        else:
            G = np.random.choice([0.25,1.25], size=(1,self.batch_size))
            G = np.repeat(G, 2 * self.n_in, axis=0).T

        Lambda         = G * np.exp( - (S - np.tile(self.phi, (self.batch_size, 2) ) )**2 / (2.0 * self.sigma_sq) )
        R              = np.random.poisson(Lambda)

        example_input  = R
        resp_1         = R[:,0:self.n_in]
        resp_2         = R[:,self.n_in:]
        
        example_input  = np.concatenate((resp_1,resp_2), axis=1)
               
        example_output = ST.T
        opt_s          = np.zeros((self.batch_size,))
        
        # calculate posterior mean
        for i in range(self.batch_size):
            pr = np.zeros((self.ngp,self.ngp))            
            for j in range(self.n_in):
                pr = pr - R[i,j] * (self.ss1 - self.phis[j] )**2 / (2.0 * self.sigma_sq) - G[i,j] * np.exp( - (self.ss1 - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            for j in range(self.n_in,2*self.n_in):
                pr = pr - R[i,j] * (self.ss2 - self.phis[j] )**2 / (2.0 * self.sigma_sq) - G[i,j] * np.exp( - (self.ss2 - self.phis[j] )**2 / (2.0 * self.sigma_sq) ) 
            
            pr       = np.exp(pr)
            pr       = pr / np.sum(pr)
            opt_s[i] = np.dot(pr.flatten(),self.ss.flatten())

        return example_input, example_output, G[:,0], G[:,-1], example_output, np.expand_dims(opt_s, axis=1)


class KalmanFilteringTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigtc_sq=4.0, signu_sq=1.0, gamma=0.1, tr_cond='all_gains'):
        super(KalmanFilteringTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.stim_dur   = stim_dur
        self.sigtc_sq   = sigtc_sq
        self.signu_sq   = signu_sq
        self.gamma      = gamma
        self.tr_cond    = tr_cond
        self.phi        = np.concatenate((np.linspace(-9.0, 9.0, 0.8*self.n_in),np.linspace(-9.0, 9.0, np.ceil((1.0-0.8)*self.n_in))))

    def sample(self):
        
        NU         = np.sqrt(self.signu_sq) * np.random.randn(self.stim_dur, self.batch_size)
        R          = np.zeros((self.n_in, self.stim_dur, self.batch_size))
        S          = np.zeros((1, self.stim_dur, self.batch_size))
        M          = np.zeros((1, self.stim_dur, self.batch_size))        
        SIG_SQ     = np.zeros((1, self.stim_dur, self.batch_size))
        M_IN       = np.zeros((1, self.stim_dur, self.batch_size))
        SIG_SQ_IN  = np.zeros((1, self.stim_dur, self.batch_size))
        
        A_in       = np.ones((1, self.n_in))
        B_in       = self.phi
        
        if self.tr_cond == 'all_gains':
            G         = (3.0 - 0.3) * np.random.rand(self.stim_dur, self.batch_size) + 0.3
        else:
            G         = np.random.choice([0.3, 3.0],(self.stim_dur, self.batch_size))

        for ii in range(self.batch_size):
            S[0,0,ii]         = np.sqrt(self.signu_sq) * np.random.randn()
            R[:,0,ii]         = G[0,ii] * np.exp(- ((S[0,0,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
            R[:,0,ii]         = np.random.poisson(R[:,0,ii])
            M[0,0,ii]         = np.dot(B_in, R[:,0,ii]) / ( np.dot(A_in, R[:,0,ii]) + (self.sigtc_sq/self.signu_sq))
            SIG_SQ[0,0,ii]    = 1.0 / ( np.dot(A_in, R[:,0,ii]) / self.sigtc_sq + (1.0 / self.signu_sq))
            M_IN[0,0,ii]      = M[0,0,ii]
            SIG_SQ_IN[0,0,ii] = SIG_SQ[0,0,ii]
            
            for tt in range(1,self.stim_dur):
                
                S[0,tt,ii]         = (1.0 - self.gamma) * S[0,tt-1,ii] + NU[tt,ii]
                R[:,tt,ii]         = G[tt,ii] * np.exp(- ((S[0,tt,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
                R[:,tt,ii]         = np.random.poisson(R[:,tt,ii])
               
                natparam_1_in      = np.dot(B_in, R[:,tt,ii]) / self.sigtc_sq
                natparam_2_in      = np.dot(A_in, R[:,tt,ii]) / self.sigtc_sq
               
                M_IN[0,tt,ii]      = natparam_1_in / natparam_2_in
                SIG_SQ_IN[0,tt,ii] = 1.0 / natparam_2_in
               
                K                  = self.signu_sq + (1.0-self.gamma)**2 * SIG_SQ[0,tt-1,ii]
                       
                M[0,tt,ii]         = ( np.dot(B_in, R[:,tt,ii]) * K + (1.0-self.gamma) * M[0,tt-1,ii] * self.sigtc_sq) / ( np.dot(A_in, R[:,tt,ii]) * K + self.sigtc_sq)
                SIG_SQ[0,tt,ii]    = (self.sigtc_sq * K) / (np.dot(A_in, R[:,tt,ii]) * K + self.sigtc_sq)
            
        example_input         = np.swapaxes(R,0,2)
        example_output        = np.swapaxes(S,0,2)
        opt_s                 = np.swapaxes(M,0,2)

        return example_input, example_output, opt_s
        

class KalmanFilteringTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=1, n_in=50, n_out=1, stim_dur=10, sigtc_sq=4.0, signu_sq=1.0, gamma=0.1, tr_cond='all_gains'):
        super(KalmanFilteringTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.batch_size = batch_size        
        self.n_in       = n_in
        self.n_out      = n_out
        self.stim_dur   = stim_dur
        self.sigtc_sq   = sigtc_sq
        self.signu_sq   = signu_sq
        self.gamma      = gamma
        self.tr_cond    = tr_cond
        self.phi        = np.linspace(-9.0, 9.0, self.n_in)

    def sample(self):
        
        NU         = np.sqrt(self.signu_sq) * np.random.randn(self.stim_dur, self.batch_size)
        R          = np.zeros((self.n_in, self.stim_dur, self.batch_size))
        S          = np.zeros((1, self.stim_dur, self.batch_size))
        M          = np.zeros((1, self.stim_dur, self.batch_size))        
        SIG_SQ     = np.zeros((1, self.stim_dur, self.batch_size))
        M_IN       = np.zeros((1, self.stim_dur, self.batch_size))
        SIG_SQ_IN  = np.zeros((1, self.stim_dur, self.batch_size))
        
        A_in       = np.ones((1, self.n_in))
        B_in       = self.phi
        
        if self.tr_cond == 'all_gains':
            G         = (3.0 - 0.3) * np.random.rand(self.stim_dur, self.batch_size) + 0.3
        else:
            G         = np.random.choice([0.3, 3.0],(self.stim_dur, self.batch_size))

        for ii in range(self.batch_size):
            S[0,0,ii]         = np.sqrt(self.signu_sq) * np.random.randn()
            R[:,0,ii]         = G[0,ii] * np.exp(- ((S[0,0,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
            R[:,0,ii]         = np.random.poisson(R[:,0,ii])
            M[0,0,ii]         = np.dot(B_in, R[:,0,ii]) / ( np.dot(A_in, R[:,0,ii]) + (self.sigtc_sq/self.signu_sq))
            SIG_SQ[0,0,ii]    = 1.0 / ( np.dot(A_in, R[:,0,ii]) / self.sigtc_sq + (1.0 / self.signu_sq))
            M_IN[0,0,ii]      = M[0,0,ii]
            SIG_SQ_IN[0,0,ii] = SIG_SQ[0,0,ii]
            
            for tt in range(1,self.stim_dur):
                
                S[0,tt,ii]         = (1.0 - self.gamma) * S[0,tt-1,ii] + NU[tt,ii]
                R[:,tt,ii]         = G[tt,ii] * np.exp(- ((S[0,tt,ii] - self.phi) / (np.sqrt(2.0 * self.sigtc_sq))) ** 2)
                R[:,tt,ii]         = np.random.poisson(R[:,tt,ii])
               
                natparam_1_in      = np.dot(B_in, R[:,tt,ii]) / self.sigtc_sq
                natparam_2_in      = np.dot(A_in, R[:,tt,ii]) / self.sigtc_sq
               
                M_IN[0,tt,ii]      = natparam_1_in / natparam_2_in
                SIG_SQ_IN[0,tt,ii] = 1.0 / natparam_2_in
               
                K                  = self.signu_sq + (1.0-self.gamma)**2 * SIG_SQ[0,tt-1,ii]
                       
                M[0,tt,ii]         = ( np.dot(B_in, R[:,tt,ii]) * K + (1.0-self.gamma) * M[0,tt-1,ii] * self.sigtc_sq) / ( np.dot(A_in, R[:,tt,ii]) * K + self.sigtc_sq)
                SIG_SQ[0,tt,ii]    = (self.sigtc_sq * K) / (np.dot(A_in, R[:,tt,ii]) * K + self.sigtc_sq)
            
        example_input         = np.swapaxes(R,0,2)
        example_output        = np.swapaxes(S,0,2)
        opt_s                 = np.swapaxes(M,0,2)

        return example_input, example_output, opt_s

        
class StimulusDemixingTask(Task):
    '''Parameters'''
    def __init__(self, W_mix, f_I, f_b, max_iter=None, batch_size=10, n_in=10, n_out=4, stim_dur=10, nmc = 1000, tr_cond='all_gains'):
        super(StimulusDemixingTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.n_od      = 4
        self.nneuron   = self.n_in * self.n_od # total number of input neurons
        self.stim_dur  = stim_dur
        self.nmc       = nmc
        self.tr_cond   = tr_cond
        
        # W_mix, f_I, f_b, c_min, c_max       
        self.c_min     = 2.0
        self.c_max     = 9.0
        self.W_mix     = W_mix # np.random.rand(self.n_od, self.n_out)
        self.f_I       = f_I   # np.random.rand(1,self.nneuron)
        self.f_b       = f_b   # np.random.rand(1,self.nneuron)

    def sample(self):
        
        if self.tr_cond == 'all_gains':
            C = (self.c_max - self.c_min) * np.random.rand(self.batch_size, self.n_out) + self.c_min
        else:
            C = np.repeat(np.random.choice([self.c_min + 2.0, self.c_max - 2.0], size=(self.batch_size,1)), self.n_out, axis=1)

        # Sources 
        S              = np.random.binomial(1,0.5,size=(self.batch_size, self.n_out))
        
        # Odorants 
        O              = np.dot(C * S, self.W_mix)  
        O_p            = np.kron(O, np.ones((1,self.n_in)) )
        
        # Noisy responses
        Lambda         = (1.0/self.stim_dur) * (O_p * np.repeat(self.f_I,self.batch_size,axis=0) + np.repeat(self.f_b,self.batch_size,axis=0))  
        Lambda         = np.tile(Lambda,(self.stim_dur,1,1))
        Lambda         = np.swapaxes(Lambda,0,1)

        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = np.tile(S,(self.stim_dur,1,1))
        example_output = np.swapaxes(example_output,0,1)
        
        exc1 = example_input[:,:,:int(0.8*self.n_in)]
        inh1 = example_input[:,:,int(0.8*self.n_in):self.n_in]
        exc2 = example_input[:,:,self.n_in:int(1.8*self.n_in)]
        inh2 = example_input[:,:,int(1.8*self.n_in):int(2*self.n_in)]
        exc3 = example_input[:,:,int(2*self.n_in):int(2.8*self.n_in)]
        inh3 = example_input[:,:,int(2.8*self.n_in):int(3*self.n_in)]
        exc4 = example_input[:,:,int(3*self.n_in):int(3.8*self.n_in)]
        inh4 = example_input[:,:,int(3.8*self.n_in):int(4*self.n_in)]
        
        example_input = np.concatenate((exc1,exc2,exc3,exc4,inh1,inh2,inh3,inh4),axis=2)
        
        # Actual posterior probabilitiefrom s of sources
        A0             = np.zeros( (self.batch_size, self.n_out) ) # absence
        A1             = np.zeros( (self.batch_size, self.n_out) ) # presence
        
        for i in range(self.n_out):
            for j in range(self.nmc):
                ss      = np.random.binomial(1,0.5,size=(1,self.n_out))
                ss[:,i] = 0.0
                cc      = (self.c_max - self.c_min) * np.random.rand(1, self.n_out) + self.c_min
                oo      = np.dot((cc * ss), self.W_mix)        
                op      = np.kron(oo, np.ones( (1,self.n_in) ))
                lam     = (op * self.f_I + self.f_b)  
                A0[:,i] = A0[:,i] + np.exp(np.sum( np.sum(R,axis=1) * np.log( np.tile(lam,(self.batch_size,1)) ) - np.tile(lam,(self.batch_size,1)), axis=1 ) )
                
                ss      = np.random.binomial(1,0.5,size=(1,self.n_out))
                ss[:,i] = 1.0
                cc      = (self.c_max - self.c_min) * np.random.rand(1, self.n_out) + self.c_min
                oo      = np.dot((cc * ss), self.W_mix)        
                op      = np.kron(oo, np.ones( (1,self.n_in) ))
                lam     = (op * self.f_I + self.f_b)  
                A1[:,i] = A1[:,i] + np.exp(np.sum( np.sum(R,axis=1) * np.log( np.tile(lam,(self.batch_size,1)) ) - np.tile(lam,(self.batch_size,1)), axis=1 ) )
                
        P = A1 / (A0+A1)
        
        return example_input, example_output, P


class StimulusDemixingTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, W_mix, f_I, f_b, max_iter=None, batch_size=10, n_in=10, n_out=4, nmc = 1000, tr_cond='all_gains'):
        super(StimulusDemixingTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_in      = n_in
        self.n_out     = n_out
        self.n_od      = 4
        self.nneuron   = self.n_in * self.n_od # total number of input neurons
        self.nmc       = nmc
        self.tr_cond   = tr_cond
        
        # W_mix, f_I, f_b, c_min, c_max       
        self.c_min     = 2.0
        self.c_max     = 9.0
        self.W_mix     = W_mix # np.random.rand(self.n_od, self.n_out)
        self.f_I       = f_I   # np.random.rand(1,self.nneuron)
        self.f_b       = f_b   # np.random.rand(1,self.nneuron)

    def sample(self):
        
        if self.tr_cond == 'all_gains':
            C = (self.c_max - self.c_min) * np.random.rand(self.batch_size, self.n_out) + self.c_min
        else:
            C = np.repeat(np.random.choice([self.c_min + 2.0, self.c_max - 2.0], size=(self.batch_size,1)), self.n_out, axis=1)

        # Sources 
        S              = np.random.binomial(1,0.5,size=(self.batch_size, self.n_out))
        
        # Odorants 
        O              = np.dot(C * S, self.W_mix)  
        O_p            = np.kron(O, np.ones((1,self.n_in)) )
        
        # Noisy responses
        Lambda         = (O_p * np.repeat(self.f_I,self.batch_size,axis=0) + np.repeat(self.f_b,self.batch_size,axis=0))  

        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = S
        
        # Actual posterior probabilitiefrom s of sources
        A0             = np.zeros( (self.batch_size, self.n_out) ) # absence
        A1             = np.zeros( (self.batch_size, self.n_out) ) # presence
        
        for i in range(self.n_out):
            for j in range(self.nmc):
                ss      = np.random.binomial(1,0.5,size=(1,self.n_out))
                ss[:,i] = 0.0
                cc      = (self.c_max - self.c_min) * np.random.rand(1, self.n_out) + self.c_min
                oo      = np.dot((cc * ss), self.W_mix)        
                op      = np.kron(oo, np.ones( (1,self.n_in) ))
                lam     = (op * self.f_I + self.f_b)  
                A0[:,i] = A0[:,i] + np.exp(np.sum( R * np.log( np.tile(lam,(self.batch_size,1)) ) - np.tile(lam,(self.batch_size,1)), axis=1 ) )
                
                ss      = np.random.binomial(1,0.5,size=(1,self.n_out))
                ss[:,i] = 1.0
                cc      = (self.c_max - self.c_min) * np.random.rand(1, self.n_out) + self.c_min
                oo      = np.dot((cc * ss), self.W_mix)        
                op      = np.kron(oo, np.ones( (1,self.n_in) ))
                lam     = (op * self.f_I + self.f_b)  
                A1[:,i] = A1[:,i] + np.exp(np.sum( R * np.log( np.tile(lam,(self.batch_size,1)) ) - np.tile(lam,(self.batch_size,1)), axis=1 ) )
                
        P = A1 / (A0+A1)
        
        return example_input, example_output, P


class VisualSearchTaskFFWD(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=10, n_loc=4, n_in=20, n_out=1, kappa=2.0, sT=0.0, n_rm=1000, tr_cond='all_gains'):
        super(VisualSearchTaskFFWD, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_rm      = n_rm        
        self.n_in      = n_in                   # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.sT        = sT
        self.nneuron   = self.n_in * self.n_loc # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in)
        self.Z         = np.linspace(0, np.pi, self.n_rm)
        self.dz        = np.pi/self.n_rm
        self.tr_cond   = tr_cond
        
    def sample(self):
        
        if self.tr_cond == 'all_gains':
            G = np.random.choice([0.5, 3.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
        else:
            G = np.random.choice([0.5, 3.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            
        # Target presence/absence and stimuli 
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size,1))
        C1ind          = np.where(C==1.0)[0]        

        S              = np.pi * np.random.rand(self.n_loc, self.batch_size)
        S[np.random.randint(0,self.n_loc,size=(len(C1ind),)), C1ind] = self.sT 
        S              = np.repeat(S,self.n_in,axis=0).T
                
        # Noisy responses
        Lambda         = G * np.exp( self.kappa * (np.cos( 2.0 * (S - np.tile(self.phi, (self.batch_size,self.n_loc) ) ) ) - 1.0) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = C
        
        # Actual posterior probabilities 
        SECOND_TERM    = np.zeros( (self.batch_size, self.n_loc) )
        
        RSUM           = np.reshape( R, (self.batch_size, self.n_loc, self.n_in) ) 

        for i in range(self.n_rm):
            SECOND_TERM = SECOND_TERM + np.exp( self.kappa * np.sum( RSUM * (np.cos( 2.0 * (self.Z[i] - np.tile(self.phi, (self.batch_size, self.n_loc, 1) ) ) ) - 1.0), axis=2 ) )
        
        SECOND_TERM = np.log(SECOND_TERM) + np.log(self.dz) - np.log(np.pi) 
        FIRST_TERM  = self.kappa * np.sum( RSUM * (np.cos( 2.0 * (self.sT - np.tile(self.phi, (self.batch_size, self.n_loc, 1) ) ) ) - 1.0), axis=2 ) 
        D_I         = FIRST_TERM - SECOND_TERM 
        D           = np.log( np.sum( np.exp(D_I), axis=1 ) ) - np.log(self.n_loc)
        P           = 1.0 / (1.0 + np.exp(-D))

        return example_input, example_output, P


class VisualSearchTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=10, n_loc=4, n_in=20, n_out=1, stim_dur=10, kappa=2.0, sT=0.0, n_rm=1000, tr_cond='all_gains'):
        super(VisualSearchTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_rm      = n_rm        
        self.n_in      = n_in                   # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.sT        = sT
        self.nneuron   = self.n_in * self.n_loc # total number of input neurons
        self.phi       = np.concatenate((np.linspace(0, np.pi, 0.8*self.n_in),np.linspace(0, np.pi, np.ceil((1.0-0.8)*self.n_in)))) # np.linspace(0, np.pi, self.n_in)
        self.Z         = np.linspace(0, np.pi, self.n_rm)
        self.dz        = np.pi/self.n_rm
        self.stim_dur  = stim_dur
        self.tr_cond   = tr_cond
        
    def sample(self):
        
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([0.5, 3.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (1.0/self.stim_dur) * np.random.choice([0.5, 3.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
            
        # Target presence/absence and stimuli 
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size, 1))
        C1ind          = np.where(C==1.0)[0]        

        S              = np.pi * np.random.rand(self.n_loc, self.batch_size)
        S[np.random.randint(0,self.n_loc,size=(len(C1ind),)), C1ind] = self.sT 
        S              = np.repeat(S,self.n_in,axis=0).T
        S              = np.tile(S,(self.stim_dur,1,1))
        S              = np.swapaxes(S,0,1)
                
        # Noisy responses
        Lambda         = G * np.exp( self.kappa * (np.cos( 2.0 * (S - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = np.repeat(C,self.stim_dur,axis=1)
        example_output = np.expand_dims(example_output,axis=2)
        
        exc1 = example_input[:,:,:int(0.8*self.n_in)]
        inh1 = example_input[:,:,int(0.8*self.n_in):self.n_in]
        exc2 = example_input[:,:,self.n_in:int(1.8*self.n_in)]
        inh2 = example_input[:,:,int(1.8*self.n_in):int(2*self.n_in)]
        exc3 = example_input[:,:,int(2*self.n_in):int(2.8*self.n_in)]
        inh3 = example_input[:,:,int(2.8*self.n_in):int(3*self.n_in)]
        exc4 = example_input[:,:,int(3*self.n_in):int(3.8*self.n_in)]
        inh4 = example_input[:,:,int(3.8*self.n_in):int(4*self.n_in)]
        
        example_input = np.concatenate((exc1,exc2,exc3,exc4,inh1,inh2,inh3,inh4),axis=2)

        # Actual posterior probabilities 
        SECOND_TERM    = np.zeros( (self.batch_size, self.n_loc) )
        
        RSUM           = np.reshape( np.sum(R,axis=1), (self.batch_size, self.n_loc, self.n_in) ) 

        for i in range(self.n_rm):
            SECOND_TERM = SECOND_TERM + np.exp( self.kappa * np.sum( RSUM * (np.cos( 2.0 * (self.Z[i] - np.tile(self.phi, (self.batch_size, self.n_loc, 1) ) ) ) - 1.0), axis=2 ) )
        
        SECOND_TERM = np.log(SECOND_TERM) + np.log(self.dz) - np.log(np.pi) 
        FIRST_TERM  = self.kappa * np.sum( RSUM * (np.cos( 2.0 * (self.sT - np.tile(self.phi, (self.batch_size, self.n_loc, 1) ) ) ) - 1.0), axis=2 ) 
        D_I         = FIRST_TERM - SECOND_TERM 
        D           = np.log( np.sum( np.exp(D_I), axis=1 ) ) - np.log(self.n_loc)
        P           = 1.0 / (1.0 + np.exp(-D))

        return example_input, example_output, P


class CurrVisualSearchTask(Task):
    '''Parameters'''
    def __init__(self, max_iter=None, batch_size=10, n_loc=4, n_in=20, n_out=1, stim_dur=10, kappa=2.0, sT=0.0, n_rm=1000, max_g = 1.0, tr_cond='all_gains'):
        super(CurrVisualSearchTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.n_rm      = n_rm        
        self.n_in      = n_in                   # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.sT        = sT
        self.nneuron   = self.n_in * self.n_loc # total number of input neurons
        self.phi       = np.concatenate((np.linspace(0, np.pi, 0.8*self.n_in),np.linspace(0, np.pi, np.ceil((1.0-0.8)*self.n_in))))
        self.Z         = np.linspace(0, np.pi, self.n_rm)
        self.dz        = np.pi/self.n_rm
        self.stim_dur  = stim_dur
        self.tr_cond   = tr_cond
        self.max_g     = max_g
        
    def sample(self):
        
        if self.tr_cond == 'all_gains':
            G = (self.max_g/self.stim_dur) * np.random.choice([0.5, 3.0], size=(self.n_loc,self.batch_size))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (self.max_g/self.stim_dur) * np.random.choice([0.5, 3.0], size=(1,self.batch_size))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
            
        # Target presence/absence and stimuli 
        C              = np.random.choice([0.0, 1.0], size=(self.batch_size, 1))
        C1ind          = np.where(C==1.0)[0]        

        S              = np.pi * np.random.rand(self.n_loc, self.batch_size)
        S[np.random.randint(0,self.n_loc,size=(len(C1ind),)), C1ind] = self.sT 
        S              = np.repeat(S,self.n_in,axis=0).T
        S              = np.tile(S,(self.stim_dur,1,1))
        S              = np.swapaxes(S,0,1)
                
        # Noisy responses
        Lambda         = G * np.exp( self.kappa * (np.cos( 2.0 * (S - np.tile(self.phi, (self.batch_size,self.stim_dur,self.n_loc) ) ) ) - 1.0) )
        R              = np.random.poisson(Lambda)
        example_input  = R
        example_output = np.repeat(C,self.stim_dur,axis=1)
        example_output = np.expand_dims(example_output,axis=2)
        
        exc1 = example_input[:,:,:int(0.8*self.n_in)]
        inh1 = example_input[:,:,int(0.8*self.n_in):self.n_in]
        exc2 = example_input[:,:,self.n_in:int(1.8*self.n_in)]
        inh2 = example_input[:,:,int(1.8*self.n_in):int(2*self.n_in)]
        exc3 = example_input[:,:,int(2*self.n_in):int(2.8*self.n_in)]
        inh3 = example_input[:,:,int(2.8*self.n_in):int(3*self.n_in)]
        exc4 = example_input[:,:,int(3*self.n_in):int(3.8*self.n_in)]
        inh4 = example_input[:,:,int(3.8*self.n_in):int(4*self.n_in)]
        
        example_input = np.concatenate((exc1,exc2,exc3,exc4,inh1,inh2,inh3,inh4),axis=2)        
        
        # Actual posterior probabilities 
        SECOND_TERM    = np.zeros( (self.batch_size, self.n_loc) )
        
        RSUM           = np.reshape( np.sum(R,axis=1), (self.batch_size, self.n_loc, self.n_in) ) 

        for i in range(self.n_rm):
            SECOND_TERM = SECOND_TERM + np.exp( self.kappa * np.sum( RSUM * (np.cos( 2.0 * (self.Z[i] - np.tile(self.phi, (self.batch_size, self.n_loc, 1) ) ) ) - 1.0), axis=2 ) )
        
        SECOND_TERM = np.log(SECOND_TERM) + np.log(self.dz) - np.log(np.pi) 
        FIRST_TERM  = self.kappa * np.sum( RSUM * (np.cos( 2.0 * (self.sT - np.tile(self.phi, (self.batch_size, self.n_loc, 1) ) ) ) - 1.0), axis=2 ) 
        D_I         = FIRST_TERM - SECOND_TERM 
        D           = np.log( np.sum( np.exp(D_I), axis=1 ) ) - np.log(self.n_loc)
        P           = 1.0 / (1.0 + np.exp(-D))

        return example_input, example_output, P
