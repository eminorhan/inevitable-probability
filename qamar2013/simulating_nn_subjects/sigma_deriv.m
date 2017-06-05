function s = sigma_deriv(h, ftype)

if strcmp(ftype,'tanh')
    s = 1-tanh(h).^2;
elseif strcmp(ftype,'relu')
    s = heaviside(h);
elseif strcmp(ftype,'sigm')
    s = 0.25*(1-tanh(h/2).^2);    
end