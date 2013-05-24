function allCodeVecs = compute_all_code_vectors(voc, classes, imagePath, cache)
%
    if nargin < 3
        imagePath = 'data/myImages/';
        cache = 'data/cache/global/';
    end
    if nargin < 4
        cache = 'data/cache/global/';
    end

    allCodeVecs = cell(numel(classes),1);
    for i = 1:numel(classes)
        names = classes{i}.images;
        allCodeVecs{i} = computeCodeVectorFromImageList(voc,names, ...
            [cache,classes{i}.name], [imagePath,classes{i}.name]);
    end

end