function [wall, allLabels] = freq_gen(allHists, classes, numTrainPerClass, ...
    numVisWords, numCls, numRegions)
% generative model for learning a BOW model
% [OUTPUTS]
% wall: model for all classes
% allLabels: annotation info, training and testing info
% [INPUTS]
% allHists: hists in a cell array, each element stores a hist for one image
% classes: annotations for all classes

    % fixed model for each region
    wall = 1/numVisWords .* ones(numRegions, numVisWords, numCls);
    
    % select training images, this step should be randomly do 5 times
    trainset = [];
    trainset = logical(trainset);
    labels = [];
    for i = 1:numel(classes)
        %numTrain = round(length(classes{i}.images) * trainPortion);
        numTrain = numTrainPerClass;
        index = randperm(length(classes{i}.images));
        trainidx = index(1:numTrain);
        temp = ismember(1:length(classes{i}.images), trainidx);
        trainset = cat(2, trainset, temp);
        
        labels = cat(1, labels, i.*ones(length(classes{i}.images),1));
    end   
    allLabels = zeros(length(trainset),2);
    allLabels(:,1) = labels;
    allLabels(:,2) = ~trainset; % note: the output for testing
    
    % original Hists without been normalized or processed
    Hists = [allHists{:}]';
    trainHists = Hists(trainset,:);
    trainlabels = labels(trainset);
    
    for y = 1:numCls
        wTemp = trainHists(y==trainlabels,:);
        wTemp = sum(wTemp, 1);

        wTemp = reshape(wTemp,numVisWords,numRegions);
        wTemp = wTemp + wall(:,:,y)';
        wTemp = wTemp ./ repmat(sum(wTemp,1),numVisWords,1);
        wall(:,:,y) = wTemp';
    end

end

function ck = observe_words(k, numImg, numRegions, numVisWords, Hists)
% ck is a I-by-R matrix for kth bin of a model
% which means the observation that how many kth visual words are seen 
% in image i and region r
    ck = zeros(numImg, numRegions);
    for i = 1:numImg
        for r = 1:numRegions
            ck(i,r) = Hists(i,(r-1)*numVisWords+k);
        end
    end
end
