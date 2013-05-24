%%-------------------------------------------------------------------------
% plsa or lda 
%%-------------------------------------------------------------------------
dbstop if error

if ~exist('PATH_OK')
    addpath funcs/
    setup
    PATH_OK = 1;
end

if ~exist( 'allHists', 'var')
    %% make annotations
    classes = make_anno('./indoor_data/myImages');

    %% create vocabulary or load from ./data/cache/global
    voc = computeVocabularyFromImageList(classes,'indoor_data');

    %% compute histograms of kth words seen in regions
    % control partition in computeHistogram
    allHists = compute_all_hists(voc, classes, 'indoor_data/cache/global/');
end

%% 
% select training images, this step should be randomly do 5 times
trainset = [];
trainset = logical(trainset);
labels = [];
numTrain = 100;
for i = 1:numel(classes)
    %numTrain = round(length(classes{i}.images) * trainPortion);    
    index = randperm(length(classes{i}.images));
    trainidx = index(1:numTrain);
    temp = ismember(1:length(classes{i}.images), trainidx);
    trainset = cat(2, trainset, temp);        
    labels = cat(1, labels, i.*ones(length(classes{i}.images),1));
end   
allLabels = zeros(length(trainset),2);
allLabels(:,1) = labels;
allLabels(:,2) = ~trainset; % note: the output for testing

Hists = [allHists{:}]';
testset = logical(allLabels(:,2));
trainset = ~testset; % NOTE: allLabels(:,2) is for testing

trainHists = Hists(trainset,:);
%trainHists = trainHists ./ norm(trainHists, 1); % normalize features
labels = allLabels(:,1);
trainlabels = labels(trainset);
numTrainImg = length(trainlabels);

K = 16;
R = 16;
M = 200;
C = 67;

x = cell(numTrainImg, 1);
y = cell(numTrainImg, 1);
for i = 1:numTrainImg
    x{i} = reshape(trainHists(i,:), M, R);
    y{i} = trainlabels(i);
end

%% learning
epsilon = 0.0000001;
[wt,td,E] = plsa(x,y,R,C,epsilon,0);
a = cell(C,1);
for c = 1:C
    a{c} = zeros(K,R);
    for r = 1:R
        a{c}(:,r) = sum(td{c}(:,(r-1)*numTrain+1:r*numTrain),2);
    end
end

%% testing
testHists = Hists(testset,:);
%trainHists = trainHists ./ norm(trainHists, 1); % normalize features

testlabels = labels(testset);
numTestImg = length(testlabels);

xt = cell(numTestImg, 1);
yt = cell(numTestImg, 1);
for i = 1:numTestImg
    xt{i} = reshape(testHists(i,:), M, R);
    yt{i} = testlabels(i);
end

%[Et,td1] = plsa_test(x,y,wt,td,epsilon);
[Et,td1] = plsa_test(xt,yt,wt,td,epsilon);

accuracies = zeros(C,1);
for i = 1:C
    numImg = size(Et{i},2)/R;
    scores = zeros(C, numImg);
    for c = 1:C
        scores(c,:) = sum(reshape(Et{i}(c,:), R, numImg),1);
    end
    %Et{i} = scores;
    [~,scc] = max(scores,[],1);
    accuracies(i) = sum(scc==i)/numImg;
    fprintf('cate %d: accuracy = %f\n', i, accuracies(i));
end
fprintf('overall accuracy = %f\n', mean(accuracies));

%% demo
demo = 0;
if demo == 1
    colorModels = [0 0 1; 0 1 0; 1 0 0; 0 1 1];
    [~,ci] = max([td{:}], [], 1);

    trainNames = cell(numel(classes),1);
    for cls = 1:numel(classes)
        trainNames{cls} = classes{cls}.images;
        for p = 1:numel(trainNames{cls})
            trainNames{cls}{p} = ['data/myImages/',classes{cls}.name,'/',trainNames{cls}{p}];
        end
    end
    trainNames = cat(1,trainNames{:});
    trainNames = trainNames(trainset);

    figure(100); 
    for i = 1:30:length(trainNames)
        im = imread(trainNames{i});
        [iy,ix] = size(im);
        mask = zeros(iy,ix);
        
        % 2x2 case
        %mask(1:ceil(iy/2), 1:ceil(ix/2)) = 1;
        %mask(ceil(iy/2)+1:end, 1:ceil(ix/2)) = 2;
        %mask(1:ceil(iy/2), ceil(ix/2)+1:end) = 3;
        %mask(ceil(iy/2)+1:end, ceil(ix/2)+1:end) = 4;
        
        til = R^(1/2); % R = til^2 or specify width and height
        
        for b = 1:til
            if b ~= til
                mask((b-1)*ceil(iy/til)+1:b*ceil(iy/til), 1:ceil(ix/til)) = b;
            else
                mask((b-1)*ceil(iy/til)+1:end, 1:ceil(ix/til)) = b;
            end
        end
        %mask(1:ceil(iy/til), 1:ceil(ix/til)) = block;
        %mask(ceil(iy/4)+1:2*ceil(iy/4), 1:ceil(ix/4)) = 2;
        %mask(2*ceil(iy/4)+1:3*ceil(iy/4), 1:ceil(ix/4)) = 3;
        %mask(3*ceil(iy/4)+1:end, 1:ceil(ix/4)) = 4;
        
        %mask(1:ceil(iy/4), ceil(ix/4)+1:2*ceil(ix/4)) = 5;
        %mask(ceil(iy/4)+1:2*ceil(iy/4), ceil(ix/4)+1:2*ceil(ix/4)) = 6;
        %mask(2*ceil(iy/4)+1:3*ceil(iy/4), ceil(ix/4)+1:2*ceil(ix/4)) = 7;
        %mask(3*ceil(iy/4)+1:end, ceil(ix/4)+1:2*ceil(ix/4)) = 8;
        
        %mask(1:ceil(iy/4), 2*ceil(ix/4)+1:3*ceil(ix/4)) = 9;
        %mask(ceil(iy/4)+1:2*ceil(iy/4), 2*ceil(ix/4)+1:3*ceil(ix/4)) = 10;
        %mask(2*ceil(iy/4)+1:3*ceil(iy/4), 2*ceil(ix/4)+1:3*ceil(ix/4)) = 11;
        %mask(3*ceil(iy/4)+1:end, 2*ceil(ix/4)+1:3*ceil(ix/4)) = 12;
        
        %mask(1:ceil(iy/4), 3*ceil(ix/4)+1:end) = 13;
        %mask(ceil(iy/4)+1:2*ceil(iy/4), 3*ceil(ix/4)+1:end) = 14;
        %mask(2*ceil(iy/4)+1:3*ceil(iy/4), 3*ceil(ix/4)+1:end) = 15;
        %mask(3*ceil(iy/4)+1:end, 3*ceil(ix/4)+1:end) = 16;

        imshow(im);       
        ind = floor(i/numTrain)*R*numTrain + mod(i,numTrain);

        fprintf('-- image %d -- ci %d %d %d %d--\n', i, ci(ind), ci(ind+100), ci(ind+200), ci(ind+300));

        %alphamask(mask == 1, colorModels(ci(ind),:), 0.3);    
        %alphamask(mask == 2, colorModels(ci(ind+100),:), 0.3);  
        %alphamask(mask == 3, colorModels(ci(ind+200),:), 0.3);  
        %alphamask(mask == 4, colorModels(ci(ind+300),:), 0.3);  
        
        

        pause(2);
    end

    figure(101);
    Nplots = 15;    
    set(gcf,'Name','Parts Distributions');
    for c = 1:C
        subplot(Nplots,1,c); imagesc(td{c});      
    end
    drawnow;

    figure(102);
    set(gcf,'Name','Parts Priors');
    for i = 1:C
        subplot(5,3,i); imagesc(a{i});
    end
    %colormap gray ;
end
