function fds = lidc_idri_dival_dataset(varargin)
if nargin >= 1
  part = varargin{1};
else
  part = 'train';
end
PATH = '/localdata/data/LIDC-IDRI/';
FILE_LIST_FILE = ['/home/jleuschn/phd/dival/datasets/lidc_idri_dival/'...
                  'lidc_idri_file_list.json'];
file_list = jsondecode(fileread(FILE_LIST_FILE));
file_list = file_list.(part);
file_list = cellfun(@(s) [PATH s], file_list, 'UniformOutput', false);
seed = 42;
if strcmp(part, 'validation')
  seed = 2;
elseif strcmp(part, 'test')
  seed = 1;
end
r = RandStream('mt19937ar', 'Seed', seed);
fds = fileDatastore(file_list, 'ReadFcn', @(f) fcn(f, r));
end

function im = fcn(filename, rand_stream)
MIN_VAL = -1024;
MAX_VAL = 3071;
im = dicomread(filename);
im = single(im(76:end-75, 76:end-75))';
info = dicominfo(filename);
im = im * info.RescaleSlope + info.RescaleIntercept;
im = im + rand_stream.rand(size(im))';
im = max(0, min(1, (im - MIN_VAL) / MAX_VAL));
end

