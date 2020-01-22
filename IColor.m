LABImages = {17};
grayImage = rgb2gray(imread('.\Dataset\Testing.png'));
for i=1:17
    filename = sprintf('%i.png',i);
    LABImages{i} = rgb2lab(imread(fullfile('.\Dataset',filename)));
    %figure(i),imshow(LABImages); to get type of variable --> disp(class(LABImages{3}));
end


KMEANS_LAB_centers = {17};
for i=1:17
    [L,centers] = imsegkmeans(uint8(LABImages{i}),16);
    KMEANS_LAB_centers{i} = centers;
end

%B = labeloverlay(LABImages{4},KMEANS_LABImages{4}); imshow(B);

%disp(numel(LABImages)); to get number of elements
%imshow(LABImages{4});

L = {17}; NumLabels = {17};
for i=1:17
    [L{i},NumLabels{i}] = superpixels(LABImages{i},50);
    %{
    figure
    BW = boundarymask(L{i});
    imshow(imoverlay(LABImages{i},BW,'yellow'),'InitialMagnification',67)
    %}
end
%{
[Lgray,NumLabelsgray] = superpixels(grayImage,25);
figure
BW = boundarymask(Lgray);
imshow(imoverlay(grayImage,BW,'yellow'),'InitialMagnification',67)

points = detectSURFFeatures(grayImage);
imshow(grayImage); hold on;
plot(points.selectStrongest(10)); 
%}

clear i; clear filename; clear BW;
