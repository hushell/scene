function [model, oriModel] = train_model(pos, neg, C)
% train SVM using libsvm
    train_labels    = [ones(1, size(pos,2)), -1 .* ones(1, size(neg,2))]';         
    train_data      = [pos, neg]';          % contains the train train_data

    % train_data =  (train_data - repmat(min(train_data,[],1),size(train_data,1),1))*spdiags(1./(max(train_data,[],1)-min(train_data,[],1))',0,size(train_data,2),size(train_data,2));

%     bestcv = 0;
%     for log2c = [0.01,0.1,1,10,100],
%         options = ['-t 0 -v 5 -c ', num2str(2^log2c)];
%         cv = svmtrain(train_labels,train_data,options);
%         if (cv >= bestcv),
%           bestcv = cv; 
%           bestc = 2^log2c; 
%           bestg = 2^log2g;
%         end
%         fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
%     end

%    cc = log2c;

    % manually tune this parameter
    cc = C;
    options=sprintf('-t 0 -c %f -b 1',cc);
    oriModel=svmtrain(train_labels,train_data,options);
    
    w = oriModel.SVs' * oriModel.sv_coef;
    b = -oriModel.rho;

    if oriModel.Label(1) == -1
      w = -w;
      b = -b;
    end
    
    model = {};
    model.w = w;
    model.b = b;

end
