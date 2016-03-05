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
r = imread('data\r.bmp');
g = imread('data\g.bmp');
b = imread('data\b.bmp');
le = imread('data\le.bmp');
fe = imread('data\fe.bmp');
nir = imread('data\nir.bmp');
%% Testing k-means for feature selection
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
    for k = 1:4
        color = img;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
    end
    
    h = figure('Name',imgarraynames{i});
    subplot(2,2,1);
    imshow(segmented_images{1}), title('objects in cluster 1');
    subplot(2,2,2);
    imshow(segmented_images{2}), title('objects in cluster 2');
    subplot(2,2,3);
    imshow(segmented_images{3}), title('objects in cluster 3');
    subplot(2,2,4);
    imshow(segmented_images{4}), title('objects in cluster 4');
    saveas(h, strcat(pwd(),'\output\figures\',imgarraynames{i},'.jpg'));
end
