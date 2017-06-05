function [R,P,S,C,K] = generate_popcode_noisy_data_allgains_10(nneuron, sig1_sq, sig2_sq, sigtc_sq)

load LeoPreparedData.mat
ndata  = length(S);

sprefs = linspace(-40,40,nneuron);

gains = (17.*K).^(-3.5) + 14;
gains = 100./(gains.*15.3524);

R1  = repmat(gains,1,nneuron) .* exp(-(repmat(S,1,nneuron) - repmat(sprefs,ndata,1)).^2 / (2*sigtc_sq));
R1  = poissrnd(R1); 
AR1 = sum(R1,2) / sigtc_sq;
BR1 = sum(R1.*repmat(sprefs,ndata,1),2) / sigtc_sq;
P1  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));

R = R1;
P = P1;
