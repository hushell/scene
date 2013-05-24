function draw_BBs(image, annotations, colorInd, contin, linWid, scores, threshold)
% IN - annotations: BBs
%     - colorInd: if set BBs will be drawn in the choosed color, otherwise
%       randomly choice. colorInd = -1 means random choice without blue
%     - contin: wether begin a new figure to draw, contin ==0 show image
%     - linWid: line width
%     - scores
%     - threshold
%
    nn = size(annotations, 1);
    win_posw =[]; win_posh = []; winw = []; winh = [];
    for i = 1:nn
        %win_posw = [win_posw; annotations(i).x];
        %win_posh = [win_posh; annotations(i).y];
        %winw = [winw; annotations(i).w];
        %winh = [winh; annotations(i).h];
        % win_posw = [win_posw; annotations(i,2)];
        % win_posh = [win_posh; annotations(i,1)];
        % winw = [winw; annotations(i,4)];
        % winh = [winh; annotations(i,3)];
        win_posw = [win_posw; annotations(i,1)];
        win_posh = [win_posh; annotations(i,2)];
        winw = [winw; annotations(i,3)-annotations(i,1)];
        winh = [winh; annotations(i,4)-annotations(i,2)];
    end
    if (nargin < 6)
        indx = 1:nn;
    else
        indx = find(scores > threshold);
    end
    
    if (nargin < 5)
        linWid = 2;
    end
    %draw the figure
    if (nargin < 4 || contin == 0)
        if (ischar(image))
            imshow(imread(image));
        else
            imshow(image);
        end
    end
    hold on;
    edge_colors={'r','g','b','c','m','y'};
    edge_colors2 = {'r','g','c','m','y'};
    eco = {};
    for i = 1:length(indx)
            ii = indx(i);
            det_rect = [win_posw(ii), win_posh(ii), winw(ii), winh(ii)];
            if (nargin < 3)
                cindx = randperm(length(edge_colors));
                eco = edge_colors;
            elseif (colorInd == -1)
                cindx = randperm(length(edge_colors2));
                eco = edge_colors2;
            else
                cindx = colorInd .* ones(length(edge_colors));
                eco = edge_colors;
            end
            rectangle('Position',det_rect,'EdgeColor',eco{cindx(1)},'LineWidth',linWid);
            if (nargin > 5)
                text(win_posw(ii),win_posh(ii),sprintf('%0.2f',scores(ii)),'Color','y');
            end
    end
%     f = getframe(gca);
%     [im, map] = frame2im(f);
%     imwrite(im, strcat(num2str(det_rect), '.png'));
end
