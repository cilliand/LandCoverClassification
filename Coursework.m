%%

%% Classification of satellite image data %%
%{
    Given 6 images files: r, g, b, nir, lidar fe, lidar le
    Objectives: 
    ? To select training samples from given source data based on information 
    in the ground truth (at least 20 samples for each class)
    
    ? To establish Gaussian models for each class with the training samples
    
    ? To apply maximum likelihood to the testing data (measured data) and classify each pixel into
    a class 
    
    ? To evaluate the classification accuracy by using a confusion matrix 
    1 – building; 2 – vegetation; 3 – car; 4 – ground
%}
%% Loading images
cd('C:\Development\Matlab\Visual Intelligence\Coursework 1\data\');
load('ground_truth.mat');
rgb = imread('data\rgb.bmp');
r = imread('data\r.bmp');
g = imread('data\g.bmp');
b = imread('data\b.bmp');
le = imread('data\le.bmp');
fe = imread('data\fe.bmp');
nir = imread('data\nir.bmp');
gts = {labelled_ground_truth, labelled_ground_truth2, labelled_ground_truth3,labelled_ground_truth4,labelled_ground_truth5,labelled_ground_truth6,labelled_ground_truth7};
%% Extract images from classified ground_truth

c = rgb2gray(rgb);
for i = 1:7
    figure;
    pixel_labels = gts{i};
    rgb_label = repmat(pixel_labels,[1 1 1]);
    segmented_images_grountTrust = cell(1,4);
    for k = 1:4
            color = c;
            color(rgb_label ~= k) = 0; %cols{k};
            segmented_images_grountTrust{k} = color;
            subplot(2,2,k);
            imshow(segmented_images_grountTrust{k});
    end
end


%% Testing k-means for feature selection
cd('C:\Development\Matlab\Visual Intelligence\Coursework 1\');
imgarray = {r,g,b,le,fe,nir};
imgarraynames = {'red','green','blue','le','fe','nir'};

for i = 1:numel(imgarray)
    img = imgarray{i};
    ab = double(img);
    ab = reshape(ab, 211*356, 1);
    [cluster_idx, cluster_center] = kmeans(ab,4,'distance','sqEuclidean','Replicates',3);
    pixel_labels = reshape(cluster_idx,211,356);
   
    segmented_images = cell(1,4);
    rgb_label = repmat(pixel_labels,[1 1 1]);
    
    h = figure('Name',imgarraynames{i});
    for k = 1:4
        color = img;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
        subplot(2,2,k);
        imshow(segmented_images{k}), title(sprintf('objects in cluster %d',k));
    end
        saveas(h, strcat(pwd(),'\output\figures\',imgarraynames{i},'.jpg'));
end

%%
% features = cell(6,20)
% for i = 1:numel(gts)
%     
% end
features1 = [r(26,19:38);g(26,19:38);b(26,19:38);nir(26,19:38);le(26,19:38);fe(26,19:38)];
features2 = [r(19,6:25);g(19,6:25);b(19,6:25);nir(19,6:25);le(19,6:25);fe(19,6:25)];
features3 = [r(210,29:38),r(209,29:37),r(208,30:30);g(210,29:38),g(209,29:37),g(208,30:30);b(210,29:38),b(209,29:37),b(208,30:30);nir(210,29:38),nir(209,29:37),nir(208,30:30);le(210,29:38),le(209,29:37),le(208,30:30);fe(210,29:38),fe(209,29:37),fe(208,30:30)];
features4 = [r(1:20);g(1:20);b(1:20);nir(1:20);le(1:20);fe(1:20)];
features1 = reshape(features1, [], 6);
features2 = reshape(features2, [], 6);
features3 = reshape(features3, [], 6);
features4 = reshape(features4, [], 6);


mean1 = mean(features1);
mean2 = mean(features2);
mean3 = mean(features3);
mean4 = mean(features4);

cov1 = cov(double(features1));
cov2 = cov(double(features2));
cov3 = cov(double(features3));
cov4 = cov(double(features4));
S = size(r);

    pw1 = getPdfScores(test,mean1,cov1);
    pw2 = getPdfScores(test,mean2,cov2);
    pw3 = getPdfScores(test,mean3,cov3);
    pw4 = getPdfScores(test,mean4,cov4);