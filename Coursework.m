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
%% Extract images from classified ground_truth
gts = {labelled_ground_truth, labelled_ground_truth2, labelled_ground_truth3,labelled_ground_truth4,labelled_ground_truth5,labelled_ground_truth6,labelled_ground_truth7};
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


