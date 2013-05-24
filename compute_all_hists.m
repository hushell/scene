function allHists = compute_all_hists(voc, classes, cache)
%
    if nargin < 3
        cache = 'data/cache/global/';
    end

    allHists = cell(numel(classes),1);
    for i = 1:numel(classes)
        names = classes{i}.images;
        allHists{i} = computeHistogramsFromImageList(voc,names, ...
            [cache,classes{i}.name], ['indoor_data/','myImages/',classes{i}.name]);
    end

end
