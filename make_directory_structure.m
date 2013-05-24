function []=make_directory_structure(opts)

mkdir(opts.datapath,'local')
mkdir(opts.datapath,'global')
mkdir(opts.datapath,'results')

% eventopts.localdir
mkdir(opts.globaldatapath,['local/',opts.dataset])

mkdir(opts.resdir, [opts.dataset, '/Layout'])
mkdir(opts.resdir, [opts.dataset, '/Main'])
mkdir(opts.resdir, [opts.dataset, '/Segmentation'])

[rids, gt]=textread(sprintf(opts.clsimgsetpath, ...
    opts.classes{opts.which_class},opts.trainset),'%s %d');
[eids, gt]=textread(sprintf(opts.clsimgsetpath, ...
    opts.classes{opts.which_class},opts.testset),'%s %d');

opts.image_names = cat(1, rids, eids);
opts.nimages = length(opts.image_names);

for ii=1:opts.nimages
     %mkdir(sprintf('%s/local',opts.datapath),num2string(ii,3));     
     mkdir(sprintf('%s/local',opts.datapath), opts.image_names{ii});
end
        