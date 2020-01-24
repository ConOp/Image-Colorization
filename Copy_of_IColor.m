clear;

LABImages = {17}; Images = {17};
grayImage = rgb2gray(imread('.\Dataset\Testing.png'));
for i=1:17
    filename = sprintf('%i.png',i);
    Images{i} = imread(fullfile('.\Dataset',filename));
    LABImages{i} = rgb2lab(Images{i});
    
    %figure(i),imshow(LABImages); to get type of variable --> disp(class(LABImages{3}));
end


[AllinOne,centers] = imsegkmeans(uint8(cell2mat(LABImages)),64); %labels, centers of Kmeans
img = {17};
L = {17};
NumLabels = {17};
for i=1:17
    img{i} = AllinOne(1:128,1+128*(i-1):128*i);
    [L{i},NumLabels{i}] = superpixels(img{i},128);
end


for u=1:17
    psa = zeros(max(max(L{u})),64); %stiles 64 = classes apo kmeans | seires = pli8os superpixel
    for i=1:128
        for j=1:128
            sup = L{u}(i,j); %gia ka8e eikona, gia ka8e pixel se poio superpixel anoikei
            psa(sup,img{u}(i,j)) = psa(sup,img{u}(i,j)) + 1;
        end
    end
    [max,index] = max(psa');
    clear psa; clear i; clear j;
end



firstimg(L==1);
x = regionprops(L,'PixelList');
BW = boundarymask(L);
imshow(imoverlay(firstimg,BW,'yellow'),'InitialMagnification',67);

wavelength = 20; orientation = [0 45 90 135]; g = gabor(wavelength,orientation);
outMag = imgaborfilt(firstimg,g);
K = size(outMag,3);
gaborfeatures = zeros(NumLabels,K);
for i=1:K
   res = regionprops(L,outMag(:,:,i),'MeanIntensity');
   gaborfeatures(:,i) = [res.MeanIntensity]';
end   


