clear;
colors = 16;
datasetimages=14;
superpixels_number=24;
compact=10;
LABImages = {datasetimages};
Images = {datasetimages};
grayImage = rgb2gray(imread('.\Dataset\Testing.png'));

%Initialize
for i=1:datasetimages
    filename = sprintf('%i.png',i);
    Images{i} = imread(fullfile('.\Dataset',filename));
    LABImages{i} = rgb2lab(Images{i});
    
    %figure(i),imshow(LABImages); to get type of variable --> disp(class(LABImages{3}));
end


%KMEANS
[AllinOne,centers] = imsegkmeans(uint8(cell2mat(LABImages)),colors); %labels, centers of Kmeans
img = {datasetimages};
L = {datasetimages};
NumLabels = {datasetimages};
%SUPERPIXELS
for i=1:datasetimages
    img{i} = AllinOne(1:128,1+128*(i-1):128*i);
    [L{i},NumLabels{i}] = superpixels(LABImages{i},superpixels_number,'Compactness',compact,'IsInputLab',true);
end

num_superpixels = 0;
for i=1:datasetimages    
    num_superpixels = num_superpixels + max(max(L{i}));
    clear max;
end
%y fro predict
y = strings([num_superpixels,1]);
index = {datasetimages};
adder = 0;
%classes
for u=1:datasetimages
    psa = zeros(max(max(L{u})),colors); %stiles 64 = classes apo kmeans | seires = pli8os superpixel
    for i=1:128
        for j=1:128
            sup = L{u}(i,j); %gia ka8e eikona, gia ka8e pixel se poio superpixel anoikei
            psa(sup,img{u}(i,j)) = psa(sup,img{u}(i,j)) + 1;
        end
    end
    [max,index{u}] = max(psa');
    clear max;
    for i=1:max(max(L{u}))
       y(i+adder,1) = [int2str(index{u}(i))];
    end
    adder = adder + i;
    clear max;
end
%gabor for training
wavelength = 20; orientation = [0 45 90 135]; 
g = gabor(wavelength,orientation);
gaborfeatures = {datasetimages};
for j=1:datasetimages
    outMag = imgaborfilt(rgb2gray(Images{j}),g);
    K = size(outMag,3);
    gaborfeatures{j} = zeros(NumLabels{j},K);
    for i=1:K
       res = regionprops(L{j},outMag(:,:,i),'MeanIntensity');
       gaborfeatures{j}(:,i) = [res.MeanIntensity]';
    end    
end

%Surf for training
SurfFeatures = {datasetimages};
valid_points = {datasetimages};
all_points={datasetimages};
for i=1:datasetimages
    points = detectSURFFeatures(rgb2gray(Images{i}));
    [SurfFeatures{i}, valid_points{i}] = extractFeatures(rgb2gray(Images{i}),points);
    all_points{i}=uint8(valid_points{i}.Location);
end


%test img

 gray_points = detectSURFFeatures(grayImage);
 [gsurf, gsurf_points] = extractFeatures(grayImage,gray_points);


labimg = rgb2lab( repmat(grayImage, [1 1 3]) );
[Gray_L,Gray_Num] = superpixels(grayImage,superpixels_number,'Compactness',compact);
%[Gray_L,Gray_Num] = superpixels(labimg,superpixels_number,'Compactness',compact);

outMag = imgaborfilt(grayImage,g);
K = size(outMag,3);
gray_gaborfeatures = zeros(Gray_Num,K+64);
for i=1:K
       res = regionprops(Gray_L,outMag(:,:,i),'MeanIntensity');
       gray_gaborfeatures(:,i) = [res.MeanIntensity]';
end

g_all_points=uint8(gsurf_points.Location);

for j=1:size(g_all_points,1)
        sinx=g_all_points(j,1);
        siny=g_all_points(j,2);
        gray_gaborfeatures(Gray_L(sinx,siny),5:end)=gsurf(j,:);
end


x = zeros(num_superpixels,68); %array of all superpixels in the training dataset
iterator = 0; %specifies the row on which the previous loop was
for i=1:datasetimages
    for j=1:size(gaborfeatures{i},1)
        for k=1:4
            x(iterator+j,k) = gaborfeatures{i}(j,k);
        end
    end
    iterator = iterator + size(gaborfeatures{i},1);
end



for i=1:datasetimages
    for j=1:size(all_points{i},1)
        sinx=all_points{i}(j,1);
        siny=all_points{i}(j,2);
        x(L{i}(sinx,siny),5:end)=SurfFeatures{i}(j,:);
    end
    
end

fprintf("Creating model ...\n");
model = fitcecoc(x,y);
saveCompactModel(model, 'tenimageseightcolors');
result = predict(model,gray_gaborfeatures);
imgrecreated = zeros(128,128,3);
%flag = false;
for i=1:size(result)
    for j=1:128
        for k=1:128
            if isequal(Gray_L(j,k), i)
                       imgrecreated(j,k,:)= centers(str2double(result(i)),:);
                       %flag=true;
                    %elseif flag == true
                       % flag=false;
                      % break;

           end
            
        end
    end
    fprintf(int2str(i));
end
a =lab2rgb(imgrecreated,'OutputType','uint8');
BW = boundarymask(Gray_L);
b=imoverlay(grayImage,BW,'yellow');
imshow([a,b]);




