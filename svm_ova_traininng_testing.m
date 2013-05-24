function [models, allScores, allLabels, aps] = svm_ova_traininng_testing(allHists, classes, trainPortion, drawAndTest)
%
%
    if nargin < 4
        drawAndTest = 0;
    end
    
    trainset = [];
    trainset = logical(trainset);
    labels = [];
    
    for i = 1:numel(classes)
        %trainset = 1:length(classes.images);
        numTrain = round(length(classes{i}.images) * trainPortion);
        index = randperm(length(classes{i}.images));
        trainidx = index(1:numTrain);
        temp = ismember(1:length(classes{i}.images), trainidx);
        trainset = cat(2, trainset, temp);
        
        labels = cat(1, labels, i.*ones(length(classes{i}.images),1));
    end
    
    allLabels = zeros(length(trainset),2);
    allLabels(:,1) = labels;
    allLabels(:,2) = ~trainset; % allLabels == 1 is for testing
    allScores = zeros(length(trainset),numel(classes));
    
    if strcmp(allHists, 'cell')
        Hists = [allHists{:}];
    else
        Hists = allHists;
    end
 
%     *****************    
%     % Hellinger kernel
%     Hists = sqrt(Hists);
%     % L2-norm
%     Hists = bsxfun(@times, Hists, 1./sqrt(sum(Hists.^2,1))) ;
    
    trainHists = Hists(:,trainset);
    trainLabels = labels(trainset);
    
    % for each class
    models = cell(numel(classes),1);
    C = 100 ;
    for i = 1:numel(classes)
        curLabels = -1.*ones(length(trainLabels),1);
        curLabels(trainLabels == i) = 1;
        
        % count how many images are there
        fprintf('Classifier %d:\n', i);
        fprintf('Number of training images: %d positive, %d negative\n', ...
            sum(curLabels > 0), sum(curLabels < 0)) ;
        fprintf('Number of testing images: %d positive, %d negative\n', ...
            sum(labels(~trainset) == i), sum(labels(~trainset) ~= i)) ;
        
        % training
        [w, bias] = trainLinearSVM(trainHists, curLabels, C) ;
        models{i}.w = w;
        models{i}.bias = bias;
        
        % testing
        scores = w' * Hists + bias ;
        allScores(:,i) = scores';
    end
    
    aps = zeros(numel(classes), 1);
    
    if (drawAndTest)
        for i = 1:numel(classes)
            trainNames = cell(numel(classes),1);
            for cls = 1:numel(classes)
                trainNames{cls} = classes{cls}.images;
                for p = 1:numel(trainNames{cls})
                    trainNames{cls}{p} = ['data/myImages/',classes{cls}.name,'/',trainNames{cls}{p}];
                end
            end
            trainNames = cat(1,trainNames{:});
            trainNames = trainNames(~trainset);
            
            scores = allScores(:,i);
            
            % actually it is testscores now, set scores(trainset) for
            % displaying below
            trainscores = scores(~trainset);
            
            %figure(100) ; clf ; set(100,'name',['Ranked test images:',classes{i}.name]) ;
            %displayRankedImageList(trainNames, trainscores)  ;
            %pause();
            
            testLabels = labels(~trainset);
            testLabels(testLabels ~= i) = -1;
            testLabels(testLabels == i) = 1;
            
            %figure(101) ; clf ; set(101,'name',['Precision-recall:',classes{i}.name]) ;
            %vl_pr(testLabels, trainscores) ;
            %pause();  
            
            % Print results
            % TODO: understand how evaluation works here
            disp('------------------------------------------------------');
            [drop,drop,info] = vl_pr(testLabels, trainscores) ;
            aps(i) = info.auc;
            fprintf('%s Test AP: %.2f\n', classes{i}.name, info.auc) ;

            [drop,perm] = sort(trainscores,'descend') ;
            fprintf('%s Correctly retrieved in the top 36: %d\n', classes{i}.name, sum(testLabels(perm(1:36)) > 0)) ;
        end
    end
end
