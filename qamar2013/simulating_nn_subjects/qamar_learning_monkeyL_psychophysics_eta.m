function qamar_learning_monkeyL_psychophysics_eta(int_indx)

rng(int_indx);

% [GAMMAE,ETA0] = meshgrid(logspace(-6,-4,6),logspace(-2.5,-1.5,6));
% GAMMAE        = GAMMAE(:);
% ETA0          = ETA0(:);

% network parameters
nhu       = 200;
L         = 3;
nneuron   = 50;
nnode     = [nneuron nhu 1];
ftype     = 'relu';
objective = 'xent';
    
% generate data
sig1_sq     = 3^2;
sig2_sq     = 15^2;
sigtc_sq    = 10^2;
[R,P,S,C,K] = generate_popcode_noisy_data_allgains_10(nneuron, sig1_sq, sig2_sq, sigtc_sq);
ndata       = length(S);
fprintf('Generated training data\n');

Xdata      = R';
Ydata      = C';

% training parameters
mu         = 0.0;
lambda_eff = 0.0;
nepch      = 1;
bsize      = 1; 
eta_0      = 0.0070;  % ETA0(int_indx);
gamma_e    = 0.0001; % GAMMAE(int_indx);
eta        = eta_0 ./ (1 + gamma_e*(0:(ndata-1))); % learning rate policy

% initialize network parameters
W_init = cell(L,1);
b_init = cell(L,1);

W_init{2} = 0.2*randn(nnode(2),nnode(1));
b_init{2} = 0.0*randn(nnode(2),1);

W_init{3} = 0.1*randn(nnode(3),nnode(2));
b_init{3} = 0.0*randn(nnode(3),1);

%% Train network with SGD
for e = 1:nepch
    
    pp = 1:ndata;
        
    % Performance over training set
    Yhattrain = zeros(1,ndata);

    for bi = 1:(ndata/bsize)
    
        bbegin = (bi-1)*bsize+1;
        bend   = bi*bsize;
        X      = Xdata(:,pp(bbegin:bend));
        Y      = Ydata(:,pp(bbegin:bend));
        
        if (e == 1) && (bi == 1)
            W = W_init;
            b = b_init;
        end
        
        for xxi = 1:size(X,2);
            [a, ~]                      = fwd_pass(X(:,xxi),W,b,L,ftype);
            Yhattrain(bbegin - 1 + xxi) = a{end};
        end

        [W, b] = do_backprop_on_batch(X, Y, W, b, eta(e), mu, lambda_eff, L, ftype, 0, objective);
   
    end

    RMSEtrain               = sqrt(mean((Yhattrain-P(pp)').^2));
    
    Yinfloss                = P(pp)';

    Yinfloss(Yhattrain==0)  = .5;
    Yhattrain(Yhattrain==0) = .5;

    InfLoss = nanmean(Yinfloss.*(log(Yinfloss./Yhattrain)) + (1-Yinfloss).*(log((1-Yinfloss)./(1-Yhattrain)))) ... 
                           / nanmean(Yinfloss.*log(2*Yinfloss) + (1-Yinfloss).*log(2*(1-Yinfloss)));
                         
    fprintf('Epoch: %i done, InfLoss on test: %f, NoAcceptedTrials: %i, RMSE on training data: %f \n', e, InfLoss, length(Yinfloss), RMSEtrain);

end

Rnet     = (Yhattrain'>0.5) + 0.0;
Snet     = S(pp);
Knet     = K(pp);
Cinf     = C(pp);

filename = strcat('qamar_psychophysics_monkeyL_run_',num2str(int_indx),'.mat');
save(filename,'Rnet','Snet','Knet','InfLoss','Cinf'); % save everything

fprintf('RUN %i DONE \n',int_indx);

end
