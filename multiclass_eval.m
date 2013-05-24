function [aps] = multiclass_eval(allScores,allLabels)
% allScores is a matrix TOTALIMG-by-NUMCLASS
% allLabels is a matrix TOTALIMG-by-2, one column is class label another is
% indicators whether is for testing
% TODO: confustion matrix
    testset = logical(allLabels(:,2)); 
    trainset = ~testset;

    trainScores = allScores(trainset,:);
    trainLabels = allLabels(:,1);
    trainLabels = trainLabels(trainset);
    [scr,idx] = max(trainScores,[],2);
    fprintf('*TRAINING AP = %f\n', sum(idx == trainLabels) ./ length(trainLabels));
    
    testScores = allScores(testset,:);
    testLabels = allLabels(:,1);
    testLabels = testLabels(testset);
    [scr,idx] = max(testScores,[],2);
    aps = sum(idx == testLabels) ./ length(testLabels);
    fprintf('*TESTING AP = %f\n', aps);
    
end
