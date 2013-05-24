function [w] = lssvm(model,x,y,lambda,epsilon,epsilon1)
% function [w] = lssvm(model,w0,x,y,lambda,epsilon,epsilon1)
%  cvpr2012_scene
%  This is a modified code from the original that takes structure arrays for x 
%  model is the current model structure that has all of the terms we need 
%  for evaluating on various parts of the method.
% [INPUT]
% model:
% w0: initial model from generative model
% x: cell array, element is MxR mat, M is len(dict), R is num(regions)
% y: 1 by N cell array, ele is class label
% lambda: trade-off between regularization and empirical error, lambda = 1/C ??
% epsilon: error bound  (e.g., 0.01)
% epsilon1: 
%
% [y,h,v] = phi_fun1(x,y,w): v = max_{y1,h1} (w * phi(x,y1,h1) + delta_fun(y,y1,h1))
% [h,v] = phi_fun2(x,y,w): v = max_{h1} w * phi(x,y,h1)
% delta_fun(y,y1,h1)
% phi(x,y,h) : argmax_{h}w * phi(x,y,h)

w = model.w;
C = model.C;
R = model.R;
M = model.M;
K = model.K;
n = length(x);
MM = (M*K + K*R); % length of wcat / C
Mw = M*K;

%%%%%% set the random number stream so we can debug easily 
%%%% this is not really needed for the final system, but is used in case a
%%%% bug is detected and we need to repeat it
[rs1] = RandStream.create('mrg32k3a','NumStreams',1,'Seed',1e7);
RandStream.setDefaultStream(rs1);          
%%%%%%

if matlabpool('size') == 0
    matlabpool 4;
end

% -- 1 -- Initialize vt and loss_t by using initial model, which can complete first guess of latent variables
wcat = []; % model.w --> wcat
for i = 1:C
    wcat = cat(1,wcat,[reshape(w{i}{1}, M*K, 1); reshape(w{i}{2}, K*R, 1)]);
end

% 1/2 lambda||w||^2
loss_t = lambda * sum(wcat.^2)/2;

if (exist('tmpLosses.mat','file'))
    fprintf('loading losses computed last time\n');
    load tmpLosses.mat;
else
    fprintf('computing initial loss_t and initial gradient\n');
    losses = zeros(1,n);
    %vts = zeros(length(w),n);
    vts = zeros(C * MM, n);
    %model.w = wcat;
    y_hPRE = struct ( 'y_',{} ...   
                    , 'h',{}  ...
                    , 'v',{}  ...
                    );
                 
    parfor i = 1:n
    %for i = 1:n
        fprintf('\t\t%02d\n',i);  
        % v = max_(y,h)[w*phi(x_i, y, h) + delta(y_i, y, h)] 
        [y_,h,v] = phi_fun1(x{i},y{i},model); % loss augmented inference   
        losses(i) = v; 
        y_hPRE(i).y_ = y_;
        y_hPRE(i).h  = h;
        y_hPRE(i).v  = v;
        %y_hPRE(i).F  = F; %ZZZ

        % latent var completion for each example
        [h,v,F] = phi_fun2(x{i}, y{i}, model);
        %g_wt(i) = v; %xx

        % Loss part of equation (5.7), v = max_h w*phi(x_i,y_i,h)
        losses(i) = losses(i) - v;
        % F = Fhi(x_i,y_i,h_i^*)
        vts(:,i) = F;
    end
end
%save('tmpLosses.mat','losses','vts','y_hPRE','g_wt');
save('tmpLosses.mat','losses','vts','y_hPRE');

% loss_t = value of equ (5.7)
loss_t = loss_t + sum(losses);

% v_t = \sum_i \Phi(x_i,y_i,h_i^*)
vt = -sum(vts,2); 

%g_wt1 = -(sum(g_wt)+ w'*vt); %xx
fprintf('\ndone computing initial loss_t %f \n',loss_t);

% ------------------------------------------------------
% Implements the CCCP method
cccp_i = 0;
%w_t1 = zeros(length(w),1); % xx
w_t1 = wcat;
while(1)
    cccp_i = cccp_i + 1;
    fprintf('outer cccp iteration %d\n',cccp_i);
    %w_t1 = bundlemethod(zeros(length(w),1), lambda, x, y, vt, epsilon1, model, y_hPRE,g_wt1); %xx

    % line 4
    % -- 2 -- standard ssvm with cutting plane algorithm
    w_t1 = bundlemethod(w_t1, lambda, x, y, vt, epsilon1, model, y_hPRE);
    fprintf('    finished bundle method\n');
    
    % -- 3 -- estimate vt again until difference of loss keep constant
    losses = zeros(1,n);
    loss_t1 = lambda * sum(w_t1.^2)/2;
    vts = zeros(C * MM, n);  % for updating the gradient
    
    %model.w = w_t1;
    for i = 1:C
        wt = w_t1((i-1)*MM+1:i*MM);
        ww = reshape(wt(1:Mw), M, K);
        aa = reshape(wt(Mw+1:end), K, R);
        model.w{i}{1} = ww;
        model.w{i}{2} = aa;
    end
    
    parfor i = 1:n
        %[y,h,v] = phi_fun1(x(i,:),y(i,:),w_t1);
        [y_,h,v] = phi_fun1(x{i},y{i},model);
        %loss_t1 = loss_t1 + v;
        losses(i) = v;
        y_hPRE(i).y_ = y_;
        y_hPRE(i).h  = h;
        y_hPRE(i).v  = v;
        %[h,v] = phi_fun2(x(i,:),y(i,:),w_t1);
        [h,v,F] = phi_fun2(x{i},y{i},model);
        %g_wt(i) = v; %xx
        %loss_t1 = loss_t1 - v;
        losses(i) = losses(i) - v;
        vts(:,i) = F;
    end
    loss_t1 = loss_t1 + sum(losses);
    vt = -sum(vts,2); 
    %g_wt1 = -(sum(g_wt)+ w_t1'*vt); %xx
    fprintf('    updated loss_t1  %f, loss_t %f\n',loss_t1,loss_t);
    
    figure(600);
    plot(w_t1);
    
    if abs(loss_t - loss_t1) < epsilon
        break;
    else
        loss_t = loss_t1;
    end

%     % save the current w!
%     w = w_t1;
%     for i = 1:C
%         %ww = w(i:i+M*K-1);
%         wt = w_t1((i-1)*MM+1:i*MM);
%         ww = reshape(wt(1:Mw), M, K);
%         %aa = w(i+M*K:i+MM-1);
%         aa = reshape(wt(Mw+1:end), K, R);
%         model.w{i}{1} = ww;
%         model.w{i}{2} = aa;
%     end
%     save('tmpW.mat','w','model');

end

w = w_t1;

matlabpool close;

end

% Improved Cutting-Plane Method for solving this convex optimization.
function w = bundlemethod(wt, lambda, x, y, vt, epsilon1, model, y_hPRE)
%function w = bundlemethod(wt,lambda, x, y, vt, epsilon1,model,y_hPRE,g_wt1) %xx
A = [];
B = [];
W = [];
%[n,d] = size(x);
n = length(y);
a = vt;

M = model.M; K = model.K; R = model.R; C= model.C;
MM = M*K + K*R; % length of part model
Mw = M*K;

%loss_t = wt' *vt + g_wt1; %xx
loss_t = wt' *vt;
losses = zeros(1,n);
as = zeros(length(wt),n);

%model.w = wt;
% wt --> model.w
for i = 1:C
    www = wt((i-1)*MM+1:i*MM);
    ww = reshape(www(1:Mw), M, K);
    aa = reshape(www(Mw+1:end), K, R);
    model.w{i}{1} = ww;
    model.w{i}{2} = aa;
end

parfor i = 1:n
%for i = 1:n
    %[y,h,v] = phi_fun1(x(i,:),y(i,:),wt);
    %a = a + phi(x(i,:),y(i,:),h);
    %[y_,h,v] = phi_fun1(x(i),y(i),model);
    %a = a + phi(x(i),y_,model,h);
    
    [v,as(:,i)] = phi(x{i}, y_hPRE(i).y_ ,model, y_hPRE(i).h);
    
    losses(i) = y_hPRE(i).v;
end
loss_t = (loss_t + sum(losses));
a = a + sum(as,2);

fprintf('bundle method: loss_t(Remp) is %f\n',loss_t);
    
b = loss_t - a'*wt;
A = [A a];
B = [B b];

qpoptions = optimset(@quadprog);
qpoptions.Algorithm = 'interior-point-convex';
qpoptions.Display = 'off';

bundlei=0;
while 1
    bundlei = bundlei + 1;
    fprintf('bundle iteration %d\n',bundlei);
    Q = A'*A;
    f = -lambda * B;
    lb = zeros(size(A,2),1);
    Aeq = ones(1,size(A,2));
    beq = 1;
    alpha = quadprog(Q,f,[],[], Aeq, beq, lb,[],[],qpoptions);
    w_t1 = -A*alpha/lambda;
    W = [W w_t1];
    a = vt;
    losses = zeros(1,n);
    %loss_t1 = w_t1' *vt +g_wt1;%xx
    loss_t1 = w_t1' *vt;
    
    %model.w = w_t1;
    % w_t1 --> model.w
    for i = 1:C
        www = w_t1((i-1)*MM+1:i*MM);
        ww = reshape(www(1:Mw), M, K);
        aa = reshape(www(Mw+1:end), K, R);
        model.w{i}{1} = ww;
        model.w{i}{2} = aa;
    end
    
    as = zeros(length(w_t1),n);
    parfor i = 1:n
    %for i = 1:n
        %[y,h,v] = phi_fun1(x(i,:),y(i,:),w_t1);
        %a = a + phi(x(i,:),y(i,:),h);
        [y_,h,v] = phi_fun1(x{i},y{i},model);
        %a = a + phi(x(i),y_,model,h);
        losses(i) = v;
        
        [v,as(:,i)] = phi(x{i}, y_, model, h);
        %loss_t1 = loss_t1 + v;
        
        %ZZZ Testing
        fprintf('--- image %d ---\n', i);
        fprintf('y_ %d: ',y_);
        %fprintf('%d ',y_.spx);
        fprintf('\n');
        fprintf('yi %d: ',y{i});
        %fprintf('%d ',y(i).spx);
        fprintf('\n');
        %fprintf('    delta: %.4f\n',delta_fun(model,y(i).spx,y_.spx));
        fprintf('    loss01: %.4f\n',loss01(y{i}, y_));
    end
    loss_t1 = loss_t1 + sum(losses);
    a = a + sum(as,2);
    %w_t1  %debug
    fprintf('bundle iteration %d: loss_t1 is %f\n',bundlei,loss_t1);

    b = loss_t1 - a'*w_t1;
    A = [A a];
    B = [B b];
    J_t1 = [];
    for i = 2:size(A,2)
        temp = lambda*sum(W(:,i-1).^2)/2;
        temp1 = max(A(:,1:i)'*W(:,i-1) + B(1:i)');
        temp = temp+temp1;
        J_t1 = [J_t1 temp];
    end
    J_t = lambda*sum(w_t1.^2)/2 + max(A(:,1:end-1)'*w_t1 + B(1:end-1)');
    if abs(min(J_t1-J_t)) <= epsilon1
        break;
    end
end

%w = W(:,end-1);
w = w_t1;

fprintf('bundle returning: \n');
end


function test1(x,y,model)
% test the function f calculation
    y_.spx = randi(length(model.cof),1,length(x.spx));
    [F] = phi(x,y_,model);
    F
end

function [y_,h_,v] = phi_fun1(x,y,model)
% most violated constraint
%
% [y_,h,v] = phi_fun1(x,y,w): v = max_{y1,h1} (w * phi(x,y1,h1) + delta_fun(y,y1,h1))
%
% w is given from the model (current w)

[h_,y_,v,~] = loss_aug_inference(model,x,y);

end

function [h_,v,F] = phi_fun2(x,y,model)
% latent variable completion
%
% [h,v,phi] = phi_fun2(x,y,w): v = max_{h1} w * phi(x,y,h1)
% [INPUT]
% x is the feature mat for one example
% model is the current model -- and assume that model.w is 
% the w that we will apply

%fprintf('phi2\t');

[h_,v,F] = latent_var_completion(model, x, y, 1);

end


function [v,F] = phi(x,y,model,h)
% function [F] = phi(x,y,h,model)
%  returns the F of  w'*F, the joint feature map
%
%  model is the current model -- and assume that model.w is 
%   the w that we will apply
%
%  NOTE NOTE NOTE
%  The y inputted here is disregarded and assumed available direclty from
%  the full graph in h.

if nargin < 4
    [~,v,F] = latent_var_completion(model,x,y,1);
else
    W = model.w;
    C = model.C;
    K = model.K;
    M = model.M;
    R = model.R;
    MM = K*M + K*R;
    
    F = zeros(M,K,R);
    featPri = zeros(K,R);

    for r=1:R
        F(:,h(r),r) = x(:,r);
        featPri(h(r),r) = 1;
    end

    F = sum(F,3);
    F = reshape(F,K*M,1);
    featPri = reshape(featPri,K*R,1);
    F = cat(1,F,featPri);
    
    v = cat(1,reshape(W{y}{1},K*M,1),reshape(W{y}{2},R*K,1))' * F;
    
    F = cat(1,F,zeros((C-1)*MM,1));
    F = circshift(F,(y-1)*MM);
end

end

%===========================================================================================
function [h,v,F] = latent_var_completion(model, x, y, Fflag)
% [INPUT]
% W = {w1,...,wC}, Wy = {By,Ay}; By = is MxK; Ay is a KxR prior matrix
% x is MxR, hists for regions 
% y is fixed here
% [OUTPUT]
% F is 1xK(M+KR), which is the final feat vector for Wy, has same length as Wy 
% v = f(x,y) = max_h <W,phi(xi,yi,h)> = sum_r max_zr (<B_y_zr, phi(x_r)> + A_y_r_zr)
% h* = argmax_h <W,phi(xi,yi,h)>

C = model.C;
M = model.M;
K = model.K;
w = model.w;
Wy = w{y};
R = model.R;
vs = zeros(1,R);
h = zeros(1,R);

% sum_r max_zr (<B_y_zr, phi(x_r)> + A_y_r_zr)
for r=1:R
    %[vs(r),h(r)] = max(sum(Wy{1} .* repmat(x(:,r),1,K),1) + Wy{2}(:,r).');
    [vs(r),h(r)] = max(ones(1,M)*(Wy{1}.*repmat(x(:,r),1,K)) + Wy{2}(:,r)');
end
% f(x,y)
v = sum(vs);

% phi(x_i,y_i,h_i*) = \sum_r psi(x,r,z_r)
% psi(x,r,z_r) = [0,...,f(x_r),...,0,...,0,...,1,...,0]
if nargin >= 4 && Fflag == 1
    % feat len for a category
    MM = K*M + K*R;
    F = zeros(M,K,R);
    featPri = zeros(K,R);

    for r=1:R
        F(:,h(r),r) = x(:,r);
        featPri(h(r),r) = 1;
    end

    F = sum(F,3);
    F = reshape(F,K*M,1);
    featPri = reshape(featPri,K*R,1);
    F = cat(1,F,featPri);
    
    % (K*M+K*R)*C
    F = cat(1,F,zeros((C-1)*MM,1));
    F = circshift(F,(y-1)*MM);
end

end

function l = loss01(y,ypre)
% the feature has to be normalized with 0-1 loss

l = ~(y==ypre);

end

%===========================================================================================
function [h_,y_,v_,F] = loss_aug_inference(model,x,y)
% [INPUT]
% W = {w1,...,wC}, Wy = {By,Ay}; By = is MxK; Ay is a KxR prior matrix
% x is MxR, hists for regions 
% y is label of example, used for loss function

C = model.C;
R = model.R;
h = zeros(C,R);
v = zeros(C,1); 

M = model.M;
K = model.K;
MM = K*M + K*R;
%Fs = zeros((MM),C);

for i=1:C
   %[h(i,:),v(i),Fs(:,i)] = latent_var_completion(model,x,i,1); 
    if i == y
        v(i) = -Inf;
        continue;
    end
    [h(i,:),v(i),~] = latent_var_completion(model,x,i,1);
    v(i) = v(i) + 1;
end

[v_,y_] = max(v);
h_ = h(y_,:);

F = [];
%F = Fs(:,y_);

% (K*M+K*R)*C
%F = cat(1,F,zeros((C-1)*MM,1));
%F = circshift(F,(y-1)*MM);

end

