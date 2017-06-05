function qamar_learning_psychophysics(int_indx)

rng(int_indx);

n_data_list = floor(logspace(2,5,16)); % cumsum(repmat([72 48 48],1,5)); % 
run_list    = 1:30;

[NDATA,RUNID] = meshgrid(n_data_list,run_list);
NDATA     = NDATA(:);
RUNID     = RUNID(:);

run_idx   = RUNID(int_indx);
ndata     = NDATA(int_indx);
ndata     = ndata + (rem(ndata,2)==1); 

% network parameters
nhu       = 200;
L         = 3;
nneuron   = 50;
nnode     = [nneuron nhu 1];
ftype     = 'relu';
objective = 'xent';
    
% training parameters
mu         = 0.0;
lambda_eff = 0.0;
nepch      = 1; 
bsize      = 1; 
eta_0      = 0.05;
gamma_e    = 0.0001;
eta        = eta_0 ./ (1 + gamma_e*(0:(ndata-1))); % learning rate policy
lambda     = 0.0705; % lapse

% generate data
sig1_sq     = 3^2;
sig2_sq     = 12^2;
sigtc_sq    = 10^2;
[R,P,~,C]   = generate_popcode_simple_training(ndata, nneuron, sig1_sq, sig2_sq, sigtc_sq);
fprintf('Generated training data\n');

Xdata      = R';
Ydata      = C';

% initialize network parameters
W_init = cell(L,1);
b_init = cell(L,1);

W_init{2} = 0.05*randn(nnode(2),nnode(1));
b_init{2} = 0.00*randn(nnode(2),1);

W_init{3} = 0.05*randn(nnode(3),nnode(2));
b_init{3} = 0.00*randn(nnode(3),1);

% Evaluate network at the end of epoch
ninfloss = 12000;

%% Train network with SGD
for e = 1:nepch
    
    pp = randperm(ndata);
    
    for bi = 1:(ndata/bsize)
    
        bbegin = (bi-1)*bsize+1;
        bend   = bi*bsize;
        X      = Xdata(:,pp(bbegin:bend));
        Y      = Ydata(:,pp(bbegin:bend));
        
        if (e == 1) && (bi == 1)
            W = W_init;
            b = b_init;
        end
        
        [W, b] = do_backprop_on_batch(X, Y, W, b, eta(bi), mu, lambda_eff, L, ftype, 0, objective);
    
    end
    
    % Performance over training set
    Yhattrain           = zeros(1,ndata);
    for ti = 1:ndata
        [a, ~]          = fwd_pass(Xdata(:,ti),W,b,L,ftype);
        Yhattrain(1,ti) = a{end};
    end
    RMSEtrain = sqrt(mean((Yhattrain-P').^2));
    
    % Evaluate network at the end of epoch
    [Rinf,Pinf,Sinf,Cinf,Knet] = generate_popcode_noisy_data_allgains_6(ninfloss, nneuron, sig1_sq, sig2_sq, sigtc_sq);
    Xinfloss                   = Rinf';
    Yinfloss                   = Pinf';
    Yhatinf                    = zeros(1,ninfloss);
    for ti = 1:ninfloss
        [a, ~]        = fwd_pass(Xinfloss(:,ti),W,b,L,ftype);
        Yhatinf(1,ti) = a{end};
    end

    Yinfloss(Yhatinf==0) = .5;
    Yhatinf(Yhatinf==0)  = .5;

    InfLoss = nanmean(Yinfloss.*(log(Yinfloss./Yhatinf)) + (1-Yinfloss).*(log((1-Yinfloss)./(1-Yhatinf)))) ... 
                           / nanmean(Yinfloss.*log(2*Yinfloss) + (1-Yinfloss).*log(2*(1-Yinfloss)));
                     
    RMSE = sqrt(mean((Yhatinf-Yinfloss).^2));
    
    fprintf('Epoch: %i done, InfLoss on test: %f, RMSE on test: %f, NoAcceptedTrials: %i, RMSE on training data: %f \n', e, InfLoss, RMSE, length(Yinfloss), RMSEtrain);

end

Rnet        = (Yhatinf'>0.5) + 0.0;
Ropt        = (Yinfloss'>0.5) + 0.0;
lapses      = binornd(1,lambda,[length(Rnet),1]);
lapse_resps = (rand(length(Rnet),1)>0.5) + 0.0;
Rnet        = lapses .* lapse_resps + (1-lapses) .* Rnet + 0.0;
Snet        = Sinf;

filename = strcat('qamar_psychophysics_run_',num2str(run_idx),'_ndata_',num2str(ndata),'.mat');
save(filename,'Rnet','Snet','Knet','InfLoss','Cinf','Ropt'); % save everything

fprintf('RUN %i DONE \n',run_idx);

end
