function loglik = compute_loglik(C,S,R,which_model,params)

loglik  = 0;
ntrials = length(S);
sig1_sq = 3^2;
% 15 for Leo; 12 for Andy and humans
sig2_sq = 12^2;

alpha  = params(1);
beta   = params(2);
gamma  = params(3);
lambda = params(4);

for i = 1:ntrials
    
    c = C(i); % contrast
    s = S(i); % stimulus
    r = R(i); % response
    
    sig = sqrt( (alpha * c)^(-beta) + gamma);
    
    switch which_model
        case 'opt'
            k = sqrt( ((sig^2 + sig1_sq) * (sig^2 + sig2_sq) * log((sig^2 + sig2_sq)/(sig^2 + sig1_sq))) / (sig2_sq - sig1_sq) );
        case 'optp'
            p1 = params(5);
            k  = sqrt( ((sig^2 + sig1_sq) * (sig^2 + sig2_sq) * (log((sig^2 + sig2_sq)/(sig^2 + sig1_sq)) + 2*log(p1 / (1-p1)) ) ) / (sig2_sq - sig1_sq) );
            if ~isreal(k)
                k = 0;
            end
        case 'lin'
            k_0   = params(5);
            sig_p = params(6);
            k     = k_0 * (1 + sig/sig_p);
        case 'quad' 
            k_0   = params(5);
            sig_p = params(6);
            k     = k_0 * (1 + (sig/sig_p)^2);
        case 'fix'
            k_0 = params(5);
            k   = k_0;
    end
        
    p      = lambda * 0.5 + (1-lambda) * 0.5 * (erf((s + k) / (sig*sqrt(2))) - erf((s - k) / (sig*sqrt(2))));
    loglik = loglik + (r * log(p) + (1-r) * log(1-p)); 
    
end
