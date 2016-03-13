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
cd('data\');
load('ground_truth.mat');
rgb = imread('data\rgb.bmp');
r = imread('data\r.bmp');
g = imread('data\g.bmp');
b = imread('data\b.bmp');
le = imread('data\le.bmp');
fe = imread('data\fe.bmp');
nir = imread('data\nir.bmp');
cd('..');
imgarray = {r,g,b,le,fe,nir};
gts = {labelled_ground_truth, labelled_ground_truth2, labelled_ground_truth3,labelled_ground_truth4,labelled_ground_truth5,labelled_ground_truth6,labelled_ground_truth7};
%% Feature selection
% Programmatically select 20 features from images based on ground truth
% data.
numberOfFeatures = 20;
featureHolder = cell(1,4);
for a = 1:4
    % find index of all elements already a classified as class a
    findFeatures = find(gts{4} == a);
    % randomly sample 20 of these indices
    rng(12345); %ensure reproducability
    randSample = datasample(findFeatures,numberOfFeatures,'Replace',false);
    features = zeros(numberOfFeatures,6);
    for j = 1:numel(imgarray)
        for i = 1:numel(randSample)
            % feature at i,j = current image at the randomly select
            % sample
            features(i,j) = imgarray{j}(randSample(i));
        end
    end
    % Store feature vector
    featureHolder{a} = features;
end

% Calculate means
means = cell(1,4);
for a = 1:4
    means{a} = mean(featureHolder{a});
end

% Calculate covariances
covs = cell(1,4);
for a = 1:4
    covs{a} = cov(featureHolder{a});
end
%% Maximum Likelhood Score Calculation
    S = size(r);
    classifiedOutput = zeros(size(r));
    for i = 1:S(1)
        for j = 1:S(2)
            scores = zeros(1,4);
            pixelsToTest = zeros(1,6);
            for z = 1:numel(imgarray)
                pixelsToTest(z) = imgarray{z}(i,j);
            end
            for a = 1:4
                scores(a) = getPdfScore(pixelsToTest,means{a},covs{a});
            end
            [M, Idx] = max(scores);
            classifiedOutput(i,j) = Idx;
        end    
    end
    
%% Visualisation
c = rgb2gray(rgb);
figure;
pixel_labels = classifiedOutput;
rgb_label = repmat(pixel_labels,[1 1 1]);
segmented_classes = cell(1,4);
for k = 1:4
    color = c;
    color(rgb_label ~= k) = 0;
    segmented_classes{k} = color;
    subplot(2,2,k);
    imshow(segmented_classes{k});
end
%% Confusion matrix
target = reshape(gts{1},1,[]);
actual = reshape(classifiedOutput,1,[]);
[C,order] = confusionmat(target, actual);
accuracy = trace(C)/sum(sum(C))
C
