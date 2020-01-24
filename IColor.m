LABImages = {17}; Images = {17};
grayImage = rgb2gray(imread('.\Dataset\Testing.png'));
for i=1:17
    filename = sprintf('%i.png',i);
    Images{i} = imread(fullfile('.\Dataset',filename));
    LABImages{i} = rgb2lab(Images{i});
    
    %figure(i),imshow(LABImages); to get type of variable --> disp(class(LABImages{3}));
end


[L,centers] = imsegkmeans(uint8(cell2mat(LABImages)),16); %labels, centers of Kmeans

%B = labeloverlay(LABImages{4},KMEANS_LABImages{4}); imshow(B);

%disp(numel(LABImages)); to get number of elements
%imshow(LABImages{4});

L = {17}; NumLabels = {17}; %L label matrix of type double, and NumLabels number of superpixels that were computed.
for i=1:17
    [L{i},NumLabels{i}] = superpixels(LABImages{i},256);
    %{
    figure
    BW = boundarymask(L{i});
    imshow(imoverlay(LABImages{i},BW,'yellow'),'InitialMagnification',67)
    %}
end

%Surf feature extraction for training dataset
SurfFeatures = {17};
valid_points = {17};
for i=1:17
    points = detectSURFFeatures(rgb2gray(Images{i}));
    [SurfFeatures{i}, valid_points{i}] = extractFeatures(rgb2gray(Images{i}),points);
end
%imshow(LABImages{17}); hold on;
%plot(valid_points{17}.selectStrongest(62)); 

%Gabor feature extraction for training dataset
wavelength = 20; orientation = [0 45 90 135]; g = gabor(wavelength,orientation);
gaborfeatures = {17};
for j=1:17
    outMag = imgaborfilt(rgb2gray(Images{j}),g);
    K = size(outMag,3);
    gaborfeatures{j} = zeros(NumLabels{j},K);
    for i=1:K
       res = regionprops(L{j},outMag(:,:,i),'MeanIntensity');
       gaborfeatures{j}(:,i) = [res.MeanIntensity]';
    end    
end

%Superpixels of Test Image
[Gray_L,Gray_Num] = superpixels(grayImage,256);
%{
BW = boundarymask(L{i});
imshow(imoverlay(grayImage,BW,'yellow'),'InitialMagnification',67)
%}

%Surf feature extraction for Test Image
points = detectSURFFeatures(grayImage);
[Gray_SurfFeatures, Gray_valid_points] = extractFeatures(grayImage,points);
%{
imshow(grayImage); hold on;
plot(Gray_valid_points.selectStrongest(62)); 
%}

%Gabor feature extraction for Test Image
outMag = imgaborfilt(grayImage,g);
K = size(outMag,3);
gray_gaborfeatures = zeros(Gray_Num,K);
for i=1:K
       res = regionprops(Gray_L,outMag(:,:,i),'MeanIntensity');
       gray_gaborfeatures(:,i) = [res.MeanIntensity]';
end

%SVM
num_superpixels = 0;
for i=1:17
    num_superpixels = num_superpixels + max(max(L{i}));
end

x = zeros(num_superpixels,4); %array of all superpixels in the training dataset
iterator = 0; %specifies the row on which the previous loop was
for i=1:17
    for j=1:size(gaborfeatures{i},1)
        for k=1:4
            x(iterator+j,k) = gaborfeatures{i}(j,k);
        end
    end
    iterator = iterator + size(gaborfeatures{i},1);
end

y = repmat('Colored_Image',num_superpixels,1);

model = fitcsvm(x,y);

clear j; clear i; clear filename; clear BW;