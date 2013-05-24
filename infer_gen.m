function [probCls,aps] = infer_gen(allHists, allLabels, wall, aall, classes, sharp)
%
    if nargin < 6
        sharp = 100;
    end
    
    Hists = [allHists{:}]';
    testset = logical(allLabels(:,2)); 
    trainset = ~testset;

    labels = allLabels(:,1);

    numImg = length(trainset);
    [numRegions, numModels, numCls] = size(aall);
    numVisWords = size(wall,2);
    
    parNumImg = 4;
    if (matlabpool('size') == 0)
        matlabpool 4;
    end
    
    % compute the maximum log-likelihood
    probCls = zeros(numImg,numCls);
    
    for group = 1:parNumImg:numImg
        curNumImg = min(parNumImg,numImg-(group-1));
        parProbCls = cell(curNumImg,1);
        
        parHists = Hists(group:group+(curNumImg-1),:);
        % **************
        %parWall = repmat(wall,1,parNumImg);
        %parWall = mat2cell(parWall,numModels,numVisWords.*ones(1,parNumImg));
        %parAall = repmat(aall,1,parNumImg);
        %parWall = mat2cell(parAall,numRegions,numModels.*ones(1,parNumImg));
        % **************
        
        %for i = 1:curNumImg % seq for for debug
        parfor i = 1:curNumImg        
            fprintf('Testing image %d\n', group-1+i);
            curHist = parHists(i,:);
            pCls = zeros(1, numCls);
            for y = 1:numCls
                probRegions = zeros(1, numRegions);
                for r = 1:numRegions
                    % **************
                    %pwall = parWall(:,(i-1)*numVisWords+1:i*numVisWords);
                    %paall = parAall(:,);
                    % **************
                    
                    %probRegions(r) = ...
                    %    sum( ...
                    %    mult( ...
                    %    repmat(curHist((r-1)*numVisWords+1:r*numVisWords),numModels,1), ...
                    %    wall(:,:,y),sharp) ...
                    %    .*aall(r,:,y)',1);
                    probRegions(r) = ...
                        log_margin( ...
                        repmat(curHist((r-1)*numVisWords+1:r*numVisWords),numModels,1), ...
                        wall(:,:,y), ...
                        aall(r,:,y), ...
                        sharp ...
                        );
                end
                %pCls(y) = prod(probRegions,2);
                pCls(y) = sum(probRegions,2);
            end
            parProbCls{i} = pCls;
        end        
        probCls(group:group+(curNumImg-1),:) = cell2mat(parProbCls);
    end
    
    if (matlabpool('size') > 0)
        matlabpool close;
    end
    
    % multi-class evaluation
    trainScores = probCls(trainset,:);
    trainLabels = allLabels(:,1);
    trainLabels = trainLabels(trainset);
    [scr,idx] = max(trainScores,[],2);
    fprintf('*TRAINING AP = %f\n', sum(idx == trainLabels) ./ length(trainLabels));
    
    testScores = probCls(testset,:);
    testLabels = allLabels(:,1);
    testLabels = testLabels(testset);
    [scr,idx] = max(testScores,[],2);
    aps = sum(idx == testLabels) ./ length(testLabels);
    fprintf('*TESTING AP = %f\n', aps);
    
    % confusion matrix
    
   %% per-class performances
    draw = 0;
    if (draw)
        aps = zeros(numCls, 1);
        for i = 1:numCls
            trainNames = cell(numel(classes),1);
            for cls = 1:numel(classes)
                trainNames{cls} = classes{cls}.images;
                for p = 1:numel(trainNames{cls})
                    trainNames{cls}{p} = ['data/myImages/',classes{cls}.name,'/',trainNames{cls}{p}];
                end
            end
            trainNames = cat(1,trainNames{:});
            trainNames = trainNames(trainset);
            
            scores = probCls(:,i);
            % this is actuall testscores but I don't change name
            trainscores = scores(trainset)';
            
            figure(100) ; clf ; set(100,'name',['Ranked test images:',classes{i}.name]) ;
            displayRankedImageList(trainNames, trainscores)  ;
            pause();
            
            testscores = scores(testset);
            testLabels = labels(testset);           
            testLabels(testLabels ~= i) = -1;
            testLabels(testLabels == i) = 1;
            
            figure(101) ; clf ; set(101,'name',['Precision-recall:',classes{i}.name]) ;
            vl_pr(testLabels, trainscores) ;
            pause();  
            
            % Print results per class
            disp('------------------------------------------------------');
            [drop,drop,info] = vl_pr(testLabels, testscores) ;
            aps(i) = info.auc;
            fprintf('%s Test AP: %.2f\n', classes{i}.name, info.auc) ;

            [drop,perm] = sort(testscores,'descend') ;
            fprintf('%s Correctly retrieved in the top 36: %d\n', classes{i}.name, sum(testLabels(perm(1:36)) > 0)) ;
        end
    end
end

function prob = mult(b,v,T)
% b is a m-by-k matrix, m examples and k items in voc
% v is the weight/prob vector has the length k 
% 1/T is the sharpeness factor
    %prob = multinomial(sum(b), b)*
    %prob = mnpdf(b,v);
    
    if nargin < 3
        T = 100;
    end
    %prob = exp(1/T * log_mult(b,v));
    prob = 1/T .* log_mult(b,v);
end

function L = log_margin(x,w,a,T)
% compute one region of P(x|y) for testing using log-sum-exp trick
% x is hists with m-by-k, for this case x = repmat(x_r,numModels,1)
% w is weights with m-by-k
% T is sharpness
% a = aall(r,:,y)
    if nargin < 4
        T = 100;
    end

    logTemp = 1/T .* log_mult(x,w) + log(a');
    m = max(logTemp);
    L = m + log(sum(exp(logTemp - m),1));
end