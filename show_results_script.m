%% show results
% shows the most probable images for each of the classes

load(eventopts.trainset);
load(eventopts.testset);
load(eventopts.labels);

indexes=1:eventopts.nimages;
test_indexes=indexes(testset);

for ii=1:eventopts.nclasses
    figure(1);
    [max1,index]=sort(dec_values,'descend');
    Nplots=6;
    for jj=1:Nplots
        set(gcf,'Name',eventopts.classes{ii});
        subplot(1,Nplots,jj);imshow(read_image_db(eventopts,test_indexes(index(jj,ii)))/255);
        if(labels(test_indexes(index(jj,ii)))==ii)
            title(sprintf('prob=%f',max1(jj,ii)),'Color','g');
        else
            title(sprintf('prob=%f',max1(jj,ii)),'Color','r');
        end
    end
    drawnow
    display('press a key');
    pause
end