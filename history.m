Learn.TEM = 0;
[Pw_z2,Pz_d2,Pd2,Li] = pLSA(Xtrain,[],50,Learn);
Pq = sum(Xtest)./sum(Xtest(:));
% fold in test data
Pz_q2 = pLSA_EMfold(Xtest,Pw_z2,[],25,1);
% and output perplexity
p1 = pLSA_logL(Xtest,Pw_z,Pz_q,Pq);
p2 = pLSA_logL(Xtest,Pw_z2,Pz_q2,Pq);
p1
p2
Pq
p1
p2
Pz_q2
%-- 08/11/2012 06:04:46 PM --%
cd master/working/PLSA/
ls
cd iccv05/
ls
cd common/
ls
cd ..
ls
cd experiments/
ls
cd bag_of_words/
ls
open config_file_2.m
open config_file_1.m
pwd
cd common/
%-- 08/27/2012 06:04:29 PM --%
cd ..
cd PLSA/
ls
cd sivic/
ls
open demo_pLSA.m
open do_pLSA.m
clear
cd ../gehler/
open sampleRun.m
%-- 08/27/2012 09:27:36 PM --%
cd ../PLSA/gehler/
ls
%-- 08/27/2012 09:29:16 PM --%
cd ../PLSA/gehler/
Pw_d(:,1)
Pw_d(1,1)
%-- 08/27/2012 09:31:49 PM --%
p1
p2
clear
ls
cd ../iccv05/
ls
cd experiments/
ls
cd bag_of_words/
ls
open config_file_1.m
cd .
cd ..
cd PLSA/
cd iccv05/
cd common/
pwd
cd ../../hddc_toolbox/
cd source/
ls
open ../../gehler/sampleRun.m
mex -largeArrayDims hddc_learn_fast.c
mex -largeArrayDims hddc_learn_fast.cpp
mex -largeArrayDims hddc_learn_fast.cpp -lmwblas -lmwlapack
mex -v -largeArrayDims hddc_learn_fast.cpp -lmwblas -lmwlapack
%-- 08/28/2012 07:51:09 PM --%
open svm_ova_traininng_testing.m
open scene.m
addpath funcs/
setup
cumsum([1 2 3 4 5])
cd ~/master/working/PLSA/iccv05/
ls
cd images/
cd faces2/
cd ..
cd background_caltech/
cd ..
cd common/
[D, I] = sort(rand(10, 1))
pwd
cd ../..
cd ../
cd scene/
ls
cd ../15scene/
ls normalize.m
open normalize.m
cd ../scene/
clear
load previous.mat
open plsa.m
ls
open freq_gen.m
Hists = [allHists{:}];
[wt,td,E] = plsa(Hists,15,0,1);
cd fastdlda/
open runFastDlda.m
sum(allLabels_gen,1)
sum(allLabels_gen(1:241,:),1)
save('15scene_hists.mat', 'Hists', 'allLabels_gen');
clear
load 15scene_hists
Hists(:,1)
(~allLabels_gen(:,2))'
sum(ans(1:241))
1:6
[M,V]=size(Hists);
trainX = Hists(:,(~allLabels_gen(:,2))')';
testX = Hists(:,allLabels_gen(:,2)')';
testX = Hists(:,logical(allLabels_gen(:,2))')';
trainY = repmat(allLabels_gen(:,1),1,k);
c=15;   % 15 classes
k=15;   % 15 topics. In this example, k=c, but we can also choose k>c
trainY = repmat(allLabels_gen(:,1),1,k);
repmat(1:15,M,1);
[V,M]=size(Hists);
trainY = (trainY == repmat(1:15,M,1));
testY = trainY(:,logical(allLabels_gen(:,2))');
testY = trainY(logical(allLabels_gen(:,2))',:);
trainY = trainY(~(allLabels_gen(:,2))',:);
load data
c=15;   % 15 classes
k=15;   % 15 topics. In this example, k=c, but we can also choose k>c
[V,M]=size(Hists);
trainX = Hists(:,(~allLabels_gen(:,2))')';
testX = Hists(:,logical(allLabels_gen(:,2))')';
trainY = repmat(allLabels_gen(:,1),1,k);
trainY = (trainY == repmat(1:15,M,1));
testY = double(trainY(logical(allLabels_gen(:,2))',:));
trainY = double(trainY(~(allLabels_gen(:,2))',:));
testY = testY(:,1:14);
trainY = trainY(:,1:14);
initalpha=rand(k,1);
initbeta=[trainY,ones(M,1)-sum(trainY,2)]'*trainX;
initbeta=initbeta./(sum(initbeta,2)*ones(1,V));
lap=0.0001;
initeta=rand(k,c-1);
initbeta=[trainY,ones(M,1)-sum(trainY,2)]'*trainX;
[trainY,ones(M,1)-sum(trainY,2)]
[M,V]=size(trainX);
initalpha
initbeta=[trainY,ones(M,1)-sum(trainY,2)]'*trainX;
initbeta=initbeta./(sum(initbeta,2)*ones(1,V));
clear
imagesc(phi)
ones(15,1)
%initalpha=rand(k,1);
initalpha = ones(15,1);
initbeta=[trainY,ones(M,1)-sum(trainY,2)]'*trainX;
initbeta=initbeta./(sum(initbeta,2)*ones(1,V));
lap=0.000001;
initeta=rand(k,c-1);
% if flag=1 use the change on perplexity to check the convergence, if flag=0, use the change on parameter to check the convergence
flag=1;
[alpha,beta,eta,phi,gama,logProb_time,perplexity_time]=learnFastDlda(trainX,trainY,initalpha,initbeta,initeta,lap,flag);
[predY,accuracy,perplexity,testphi,testgama]=applyFastDlda(testX,testY,alpha,beta,eta);
imagesc(phi)
[predY,accuracy,perplexity,testphi,testgama]=applyFastDlda(trainX,trainY,alpha,beta,eta);
%-- 09/28/2012 06:53:50 AM --%
prefdir
exit
