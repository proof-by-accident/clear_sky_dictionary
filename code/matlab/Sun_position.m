clear; clc; close all;

addpath(genpath([pwd,'/utils']))
addpath(genpath([pwd,'/example_images']))

d = uigetdir(pwd,'Select directory');
fprintf('Directory: %s\n',d);

files = dir(fullfile(d,'*.jpg'));
nfiles = size(files,1);

fprintf('Reading image files...');
i = 1;
F(nfiles) = struct('cdata',[],'colormap',[]);
im = imread(files(i).name);
img = rgb2gray(im); % figure; imshowpair(im,img,'montage');
image(img);
F(i) = getframe;
[nx,ny] = size(img);
n = nx*ny;
X = zeros(n,nfiles);
X(:,i) = img(:);
for i=2:1:nfiles
    im = imread(files(i).name);
    img = rgb2gray(im);
    image(img);
    F(i) = getframe;
    X(:,i) = img(:);
end
fprintf('OK.\n');

% mrDMD
L  =  6; % number of levels
r  = 10; % rank of truncation
dt = 30; % time step
T  = nfiles*dt; % sec

fprintf('mrDMD...');
mrdmd = mrDMD(X,dt,r,2,L);
fprintf('OK.\n');

figure; imagesc(reshape(abs(mrdmd{1,1}.Phi(1:n,1)), nx, ny)); axis square;                 % level 1
for i=1:2;  figure; imagesc(reshape(abs(mrdmd{2,i}.Phi(1:n,1)), nx, ny)); axis square; end % level 2
for i=1:4;  figure; imagesc(reshape(abs(mrdmd{3,i}.Phi(1:n,1)), nx, ny)); axis square; end % level 3
% for i=1:8;  figure; imagesc(reshape(abs(mrdmd{4,i}.Phi(1:n,1)), nx, ny)); axis square; end % level 4
% for i=1:16; figure; imagesc(reshape(abs(mrdmd{5,i}.Phi(1:n,1)), nx, ny)); axis square; end % level 5
% for i=1:32; figure; imagesc(reshape(abs(mrdmd{6,i}.Phi(1:n,1)), nx, ny)); axis square; end % level 6
