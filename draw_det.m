function draw_det(image, annotations, scores, threshold)
    nn = size(annotations, 1);
    win_posw =[]; win_posh = []; winw = []; winh = [];
    for i = 1:nn
        win_posw = [win_posw; annotations(i).x];
        win_posh = [win_posh; annotations(i).y];
        winw = [winw; annotations(i).w];
        winh = [winh; annotations(i).h];
    end
    if (nargin < 3)
        indx = 1:nn;
    else
        indx = find(scores > threshold);
    end
    %draw the figure
    if (ischar(image))
        imshow(imread(image));
    else
        imshow(image);
    end
    hold on;
    edge_colors={'r','g','b','c','m','y'};
    for i = 1:length(indx)
            ii = indx(i);
            det_rect = [win_posw(ii), win_posh(ii), winw(ii), winh(ii)];
            cindx = randperm(length(edge_colors));
            rectangle('Position',det_rect,'EdgeColor',edge_colors{cindx(1)},'LineWidth',2);
            if (nargin > 2)
                text(win_posw(ii),win_posh(ii),sprintf('%0.2f',scores(ii)),'Color','y');
            end
    end
%     f = getframe(gca);
%     [im, map] = frame2im(f);
%     imwrite(im, strcat(num2str(det_rect), '.png'));
end