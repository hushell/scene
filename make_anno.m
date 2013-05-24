function [classes] = make_anno(directory)
% classes is a cell array containing name, id, and image names
% create caches for restoring features

dirs = dir(directory);
classes = cell(length(dirs),1);

mkdir([directory, '/..'], 'cache');
mkdir([directory, '/../cache'], 'global');

cnt = 1;
for i = 3:length(dirs)
   if dirs(i).isdir == 0
       continue;
   end
   
   mkdir([directory, '/../cache'], dirs(i).name);
   
   files = dir([directory, '/', dirs(i).name]);
   classes{cnt}.name = dirs(i).name;
   classes{cnt}.id = cnt;
   classes{cnt}.images = cell(length(files)-2,1);
   for j = 3:length(files)
      classes{cnt}.images{j-2} = files(j).name; 
   end
   cnt = cnt + 1;
end
classes = classes(1:cnt-1);

