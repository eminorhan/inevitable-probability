function s = sigma_func(h,ftype)

if strcmp(ftype,'tanh')
    s = tanh(h);
elseif strcmp(ftype,'relu')
    s = max(0,h);
elseif strcmp(ftype,'sigm')
    s = 0.5*tanh(h/2)+0.5;    
end