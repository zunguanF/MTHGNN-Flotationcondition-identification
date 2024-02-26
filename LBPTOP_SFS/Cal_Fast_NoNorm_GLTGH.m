clc
clear
close all
%% Fast-LBP-TOP parameter set

% radius (R=1) and the number of neighbors (P=8)
R=3;
P=8;
FxRadius = 1;                   
FyRadius = 1; 
TInterval = 1;         
NeighborPoints = [8,8,8];
TimeLength=1 ;
BorderLength = 1;
bBilinearInterpolation =1 ;
Bincount = 0;
Code = 0;
colr = 0;
tolr = 0;   
%% Mapping parameter for LBP
%       'u2'   for uniform LBP
%       'ri'   for rotation-invariant LBP
%       'riu2' for uniform rotation-invariant LBP.
patternMapping_ri = getmapping(P,'ri');
%% open csv file
filename1 = 'SFs4.csv';
fid1 = fopen(filename1, 'w');

%% import files and image data
% class Ⅳ
% class Ⅲ
% class Ⅱ
% class Ⅰ
imgDataPath = 'C:/Users/86152/Desktop/HGNN/class Ⅳ/';
imgDataDir  = dir(imgDataPath);             % 遍历所有文件
for i = 1:length(imgDataDir)
%  VolData = zeros(518, 692, 600);
% for i = 1:50

    if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                % 去除遍历中不是文件夹的
           continue;
    end
    a = dir([imgDataPath imgDataDir(i).name '/*.jpg']); 
%     for j =1:length(imgDir)                 % 遍历所有图片
%         img = imread([imgDataPath imgDataDir(i).name '/' imgDir(j).name]);
%     end
    
    for j = 1 : length(a)
    Imgdat = imread([imgDataPath imgDataDir(i).name '/' a(j).name]);
    if size(Imgdat, 3) == 3 % if color images, convert it to gray
        Imgdat = rgb2gray(Imgdat);
    end
    [height, width] = size(Imgdat);
    if j == 1
        VolData = zeros(height, width, length(a));
    end
    VolData(:, :, j) = Imgdat;
    end
    
    %% Get Fast-LBP-TOP images

    [lxy, lxt, lyt,Histogram] =LBPTOP(VolData, FxRadius, FyRadius, TInterval, NeighborPoints, TimeLength, BorderLength, bBilinearInterpolation, Bincount, Code);
    %% calculate GLCMH
    
    [height, width, Length]=size(lxy);
    % gray_leave = 36;
    % gray_leave_com = 16;
    % GLCMsequ=zeros(3,4,1);
    SFsVector=zeros(3,3);
    H=zeros(3,259);
    % dy = 2;
    % dx = 2;
    % dt = 1;
    
    % in the xy plane
    for j=1:Length
        img1=lxy(:,:,j);
        % GLCMsequ(1,:,:) = GLCMsequ(1,:,:) + GLCMH(img1,dx,dy);
        SFsVector(1,:) = SFsVector(1,:)+SFs(img1);
    end
    SFsVector(1,:)=SFsVector(1,:)/Length;
    % GLCMsequ(1,:,:)=GLCMsequ(1,:,:)/Length;
    
    % in the xt plane
    for j=1:height
        img2=reshape(lxt(j,:,:),[width,Length]);
        % GLCMsequ(2,:,:) = GLCMsequ(2,:,:) + GLCMH(img2,dx,dt);
        SFsVector(2,:) = SFsVector(2,:)+SFs(img2);
    end
    SFsVector(2,:)=SFsVector(2,:)/height;
    % GLCMsequ(2,:,:)=GLCMsequ(2,:,:)/height;

    % in the yt plane
    for j=1:width
        img3=reshape(lyt(:,j,:),[height,Length]);
        % GLCMsequ(3,:,:) = GLCMsequ(3,:,:) + GLCMH(img3,dy,dt);
        SFsVector(3,:) = SFsVector(3,:)+SFs(img3);
    end
    SFsVector(3,:)=SFsVector(3,:)/width;
    % GLCMsequ(3,:,:)=GLCMsequ(3,:,:)/width;
%% with histogram normalization in the TOP direction
%     for k=1:3
%         % H1=BHGLCMsequ;
%         for j=1:4
%             H(k,j)=GLCMsequ(k,j,1)/sum(GLCMsequ(:,j,1));
%         end
%         % H2=BSFsVector;
%         for j=1:3
%             H(k,j+4)=SFsVector(k,j)/sum(SFsVector(:,j),1);
%         end
%     end
    %% without histogram normalization in the TOP direction
    H=[SFsVector,Histogram];
    %% 写入数据
    fprintf('进度%d,总数%d\n',i,length(imgDataDir))
%     for k = 1:length(CORR)
    fprintf(fid1, '%f,', reshape(H,[1,777]));
    fprintf(fid1, '\n');
     
end
fclose ('all') ;