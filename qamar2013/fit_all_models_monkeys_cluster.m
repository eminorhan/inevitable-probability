function fit_all_models_monkeys_cluster(int_indx)

rng('shuffle');

% load LeoPreparedData.mat;
load MonkeyLData.mat;

ndataend_list      = ceil(logspace(2,5.5625,8)); 
ndataend_list(end) = 367877;

[NDATAEND,MONKEYS] = meshgrid(ndataend_list,1:50); 
NDATAEND           = NDATAEND(:);
MONKEYS            = MONKEYS(:);

ndataend           = NDATAEND(int_indx);
monkey_ind         = MONKEYS(int_indx);

% C = K(1:ndataend)';
% S = S(1:ndataend)';
% R = B(1:ndataend)';

C = Cvec(monkey_ind,1:ndataend); % CONTRAST
S = Svec(monkey_ind,1:ndataend); % STIMULI
R = Rvec(monkey_ind,1:ndataend); % RESPONSES

options = optimoptions(@fmincon,'TolFun', 1e-16, 'TolX', 1e-16,'MaxIter',1000,'display','iter');

% fit OPT
LBopt        = eps + [0 0 0 0];
UBopt        = [50 8 30 0.5];
X0opt        = [10 2 3 0.1] + 0.0333*randn(1,4);
[Xopt, Fopt] = fmincon(@(params) -compute_loglik(C,S,R,'opt',params),X0opt,[],[],[],[],LBopt,UBopt,[],options);

% fit OPT_P
LBopt_p          = eps + [0 0 0 0 .25];
UBopt_p          = [50 8 30 0.5 .75];
X0opt_p          = [10 2 3 0.1 .5] + 0.0333*randn(1,5);
[Xopt_p, Fopt_p] = fmincon(@(params) -compute_loglik(C,S,R,'optp',params),X0opt_p,[],[],[],[],LBopt_p,UBopt_p,[],options);

% fit LIN
LBlin        = eps + [0 0 0 0 0 0];
UBlin        = [50 8 30 0.5 15 50];
X0lin        = [10 2 3 0.1 3 3] + 0.0333*randn(1,6);
[Xlin, Flin] = fmincon(@(params) -compute_loglik(C,S,R,'lin',params),X0lin,[],[],[],[],LBlin,UBlin,[],options);

% fit QUAD
LBquad         = eps + [0 0 0 0 0 0];
UBquad         = [50 8 30 0.5 15 50];
X0quad         = [10 2 3 0.1 5 6] + 0.0333*randn(1,6);
[Xquad, Fquad] = fmincon(@(params) -compute_loglik(C,S,R,'quad',params),X0quad,[],[],[],[],LBquad,UBquad,[],options);

% fit FIX
LBfix        = eps + [0 0 0 0 0];
UBfix        = [50 8 30 0.5 50];
X0fix        = [10 2 3 0.1 6] + 0.0333*randn(1,5);
[Xfix, Ffix] = fmincon(@(params) -compute_loglik(C,S,R,'fix',params),X0fix,[],[],[],[],LBfix,UBfix,[],options);

NLogLikMat = [Fopt,Fopt_p,Flin,Fquad,Ffix];
fprintf('OPT: %d OPT_P: %d LIN: %d QUAD: %d FIX: %d\n',Fopt,Fopt_p,Flin,Fquad,Ffix);

filename = strcat('NLogLikMatL_monkey_',num2str(monkey_ind),'_ndata_',num2str(ndataend),'.mat');
save(filename,'NLogLikMat'); % save everything

end
