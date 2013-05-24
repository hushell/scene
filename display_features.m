function display_features(opts,detector_name,imIndex)
%Displays regions on the image
% input:
%           opts           : contains information about data set
%           detector_name  : name of detector file
%           imIndex        : image index

load(opts.image_names);
image_dir=sprintf('%s/%s/',opts.localdatapath,num2string(imIndex,3));                       % location detector
points=getfield(load(sprintf('%s/%s',image_dir,detector_name)),'points');                   % load detector

im=read_image_db(opts,imIndex);                                                             % load image

clf;imshow(im/255);

hold on;
for ii=1:length(points)
    drawcircles(points(ii,1), points(ii,2), points(ii,3),'y');                     % draw circle for all features
end
hold off

end

function drawcircles(x_c,y_c,s,col)
    t = 0:pi/50:2*pi;
    yy=s*(sin(t));
    xx=s*(cos(t));

    plot(x_c+xx,y_c+yy,'-k','LineWidth',3);
    plot(x_c+xx,y_c+yy,col,'LineWidth',1);
end


