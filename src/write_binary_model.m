RF2ArrayModel(LBFRegModel);

load ..\Models\Header.mat
load '..\Models\InitialShape_68.mat';
load ..\Models\RF.mat
load ..\Models\W.mat
    
H = single(Header);
RF = single(RF);
S0 = single(S0);
W = single(W);

precision = 'float32';
filename = '..\Models\model.bin';
fid = fopen(filename,'wb');
assert(fid>0,['打开文件' filename '失败!']);

count_header = fwrite(fid, H, precision);
count_S0 = fwrite(fid, S0', precision);
count_RF = fwrite(fid, RF', precision);
count_W = fwrite(fid, W', precision);

fclose(fid);

fprintf('write success.\n')

% B = read_binary_file(filename,precision);