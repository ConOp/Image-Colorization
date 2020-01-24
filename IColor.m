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
points = {17};
for i=1:17
    points{i} = detectSURFFeatures(rgb2gray(Images{1}));
end
imshow(LABImages{1}); hold on;
plot(points{i}.selectStrongest(62)); 

wavelength = 20; orientation = [0 45 90 135]; g = gabor(wavelength,orientation);
outMag = imgaborfilt(rgb2gray(Images{1}),g);
K = size(outMag,3);
gaborfeatures = {17}
for j=1:K
    gaborfeatures{j} = zeros(NumLabels{j},K);
    for i=1:K
       res = regionprops(L{j},outMag(:,:,i),'MeanIntensity');
       gaborfeatures{j}(:,i) = [res.MeanIntensity]';
    end    
end

clear i; clear filename; clear BW;
