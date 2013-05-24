function [wall, aall] = em_mle(allHists, allLabels, maxIter, sharp, ...
    numVisWords, numCls, numModels, numRegions)
% allHists are histograms for all images, with histograms for each region
% allLabels are labels for all images
% numIter is the number of iterations for EM algorithm

% intialize
Hists = [allHists{:}]';
trainset = logical(allLabels(:,2));
trainset = ~trainset; % NOTE: allLabels(:,2) is for testing

trainHists = Hists(trainset,:);
labels = allLabels(:,1);
trainlabels = labels(trainset);
numImg = length(trainlabels);

Q = zeros(numImg,numRegions,numModels); % i, R, L
oldQ = zeros(numImg,numRegions,numModels);

% uniform
aall = 1 ./ numModels .* ones(numRegions, numModels, numCls); % R, L, Y
wall = 1 ./ numVisWords * ones(numModels, numVisWords, numCls); % L, K, Y
% random
% aall = rand(numRegions, numModels, numCls);
% wall = rand(numModels, numVisWords, numCls);
% for r = 1:numRegions
%     for y = 1:numCls
%         aall(r,:,y) = aall(r,:,y) ./ sum(aall(r,:,y),2);
%     end
% end
% for l = 1:numModels
%     for y = 1:numCls
%         wall(l,:,y) = wall(l,:,y) ./ sum(wall(l,:,y),2);
%     end
% end

% Initial E-step: randomly assign latent variables in [1,L]
disp('* initialize Q');
Z = randi(numModels, numImg, numRegions);
% rearrange latent variables into [0,1], but each region has L latent
% variables, which is a binary vector 
for i = 1:numModels
    Q(:,:,i) = (Z == i); 
end

for it = 1:maxIter
    disp('--------------------------------------------------------------');
    fprintf('iter %d\n', it);
    
    if (it ~= 1)
        %* E-step
        tic;
        for i = 1:numImg
            for r = 1:numRegions
                posterior = ...
                    mult(trainHists(i,(r-1)*numVisWords+1:r*numVisWords), wall(Z(i,r),:,trainlabels(i)), sharp) ...
                    .* aall(r,Z(i,r),trainlabels(i)); 
                normalization = ...
                    sum( ...
                        mult( ...
                            repmat( ...
                                trainHists(i,(r-1)*numVisWords+1:r*numVisWords), ...
                                numModels,1 ...
                            ), ...
                            wall(:,:,trainlabels(i)), ...
                            sharp ...
                        ) ...
                        .* aall(r,:,trainlabels(i))', ...
                        1 ...
                    );
                posterior = posterior ./ normalization;
                Q(i,r,Z(i,r)) = posterior;
            end
        end
        fprintf('* TIME in E-step = %f\n', toc);

        % show difference between oldQ and Q
        fprintf('*** sum gain in diff(Q) = %f\n', sum(sum(sum(Q-oldQ, 3),2),1));
    end

    %* M-step
    tic;
    % update aall
    for y = 1:numCls
        for r = 1:numRegions
            for l = 1:numModels
                aall(r,l,y) = sum(Q(:,r,l) .* (trainlabels == y), 1);
            end
        end
    end

    % update wall
    for y = 1:numCls
        for l = 1:numModels
            for k = 1:numVisWords
                wall(l,k,y) = ...
                    sum( ...
                        sum(Q(:,:,l) .* observe_words( ...
                            k, numImg, numRegions, numVisWords, trainHists), ...
                        2) .* (trainlabels == y), ...
                    1);
            end
        end
    end

    % normalizations
    for y = 1:numCls
        normAOverL = sum(aall(:,:,y),2);
        normWOverK = sum(wall(:,:,y),2);
        aall(:,:,y) = aall(:,:,y) ./ repmat(normAOverL,1,numModels);
        wall(:,:,y) = wall(:,:,y) ./ repmat(normWOverK,1,numVisWords);
    end
    fprintf('* TIME in M-step = %f\n', toc);
    
    oldQ = Q;
end

end


% *** TODO pre-allocated these observed words
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
    prob = 1/T * log_mult(b,v);
end

function post = poster_model(i,r,aall,wall,Z,trainHists)
    posterior = ...
        mult( ...
            trainHists(i,(r-1)*numVisWords+1:r*numVisWords), ...
            wall(Z(i,r),:,trainlabels(i)), sharp) ...
        .* aall(r,Z(i,r),trainlabels(i)); 
    normalization = ...
        sum( ...
            mult( ...
                repmat( ...
                    trainHists(i,(r-1)*numVisWords+1:r*numVisWords), ...
                    numModels,1 ...
                ), ...
                wall(:,:,trainlabels(i)), ...
                sharp ...
            ) ...
            .* aall(r,:,trainlabels(i))', ...
            1 ...
        );
    post = posterior ./ normalization;
    %Q(i,r,Z(i,r)) = posterior;
end
