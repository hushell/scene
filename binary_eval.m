function [aps] = binary_eval(allScores, allLabels, classes)
% evaluation uses ova
% but it is not good way to evaluate generative methods which is not being treated as 
% maximum margin classifier. disadvantages:
% 1. some images have high scores or low scores for all categories. Which
%    makes some scores of right decision even lower than bad scores.
% 2. scores for different classifiers have not been calibrated.
% SOLUTION: calibrate final scores with a svm or logistic regression before evluation.

    trainset = logical(allLabels(:,2)); % ***trainset here are testset, fixed it next round
    %testHists = Hists(~trainset,:);
    labels = allLabels(:,1);
    %testlabels = labels(~trainset);
    %numImg = length(trainset);
    %[numRegions, numModels, numCls] = size(aall);
    %numVisWords = size(wall,2);
    aps = zeros(numel(classes), 1);

    draw = 1;
    if (draw)
        for i = 1:numel(classes)
%             trainNames = cell(numel(classes),1);
%             for cls = 1:numel(classes)
%                 trainNames{cls} = classes{cls}.images;
%                 for p = 1:numel(trainNames{cls})
%                     trainNames{cls}{p} = ['data/myImages/',classes{cls}.name,'/',trainNames{cls}{p}];
%                 end
%             end
%             trainNames = cat(1,trainNames{:});
%             trainNames = trainNames(trainset);
            
            scores = allScores(:,i);
            % this is actuall testscores but I don't change name
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
            disp('------------------------------------------------------');
            [drop,drop,info] = vl_pr(testLabels, trainscores) ;
            aps(i) = info.auc;
            fprintf('%s Test AP: %.2f\n', classes{i}.name, info.auc) ;

            [drop,perm] = sort(trainscores,'descend') ;
            fprintf('%s Correctly retrieved in the top 36: %d\n', classes{i}.name, sum(testLabels(perm(1:36)) > 0)) ;
        end
    end

end
