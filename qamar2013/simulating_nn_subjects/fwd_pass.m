function [a, z] = fwd_pass(h,W,b,L,ftype)

a        = cell(L,1);
z        = cell(L,1);
a{1}     = h;

for i = 1:(L-1)

    z{i+1} = W{i+1} * a{i} + b{i+1};
    if i == L-1
        a{i+1} = sigma_func(W{i+1} * a{i} + b{i+1},'sigm');
    else
        a{i+1} = sigma_func(W{i+1} * a{i} + b{i+1},ftype);
    end
end