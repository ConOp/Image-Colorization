clear;

%Variables for the colorization of the image.
%
%colors = the number of the classes for kmeans 
%(also the colors for recoloring the result image)
%
%datasetimages = the number of the images in the training set
%
%superpixels_number = the number of segments for each image
%
%compact = the compactness of the segments

colors = 128;
datasetimages=13;
superpixels_number=128;
compact=8;

%wavelength = gabor wavelength
%
%orientation = gabor degrees
wavelength = 20;
orientation = [0 45 90 135];
LABImages = {datasetimages}; %Training set images in the L*A*B* workspace
Images = {datasetimages}; %Training set images
grayImage = rgb2gray(imread('.\Dataset\Testing.png')); %The image to be colored
l_testing={datasetimages};
ab_testing={datasetimages};
[height, width] = size(grayImage);

%Initialize
for i=1:datasetimages
    filename = sprintf('%i.png',i);
    Images{i} = imread(fullfile('.\Dataset',filename));
    LABImages{i} = rgb2lab(Images{i});
    l_testing_temp = LABImages{i}(:,:,1);
    l_testing{i}=im2single(l_testing_temp);
    ab_testing_temp = LABImages{i}(:,:,2:3);
    ab_testing{i} = im2single(ab_testing_temp);
end

clearvars l_testing_temp ab_testing_temp filename;

%Kmeans on training set
[AllinOne,centers] = imsegkmeans(uint8(cell2mat(ab_testing)),colors); %labels, centers of Kmeans
img = {datasetimages};
L = {datasetimages};
NumLabels = {datasetimages};

%Superpixels
for i=1:datasetimages
    img{i} = AllinOne(1:height,1+width*(i-1):width*i); %separate kmeans labels for each image.
    [L{i},NumLabels{i}] = superpixels(LABImages{i},superpixels_number,'Compactness',compact,'IsInputLab',true);
end

num_superpixels = 0;
%Get total number of superpixels from training set
for i=1:datasetimages    
    num_superpixels = num_superpixels + max(max(L{i}));
    clear max;
end
y = strings([num_superpixels,1]); % Class label array to train SVM model
index = {datasetimages};
adder = 0;
%classes
for u=1:datasetimages
    temp_helping_array = zeros(max(max(L{u})),colors); %columns = number of kmeans centroids | rows = number of superpixels per image
    for i=1:height
        for j=1:width
            sup = L{u}(i,j); %for each image -> for each pixel get the superpixel it belongs
            % count for each superpixel the number of pixels who represent a certain centroid
            temp_helping_array(sup,img{u}(i,j)) = temp_helping_array(sup,img{u}(i,j)) + 1; 
        end
    end
    [max,index{u}] = max(temp_helping_array');
    clear max;
    for i=1:max(max(L{u}))
        %populate y, label each superpixel with its class (color)
       y(i+adder,1) = int2str(index{u}(i));
    end
    adder = adder + i;
    clear max;
end
clearvars adder temp_helping_array sup

%gabor for training set
g = gabor(wavelength,orientation);
gaborfeatures = {datasetimages};
for j=1:datasetimages
    outMag = imgaborfilt(rgb2gray(Images{j}),g);
    K = size(outMag,3);
    gaborfeatures{j} = zeros(NumLabels{j},K);
    %gabor for each superpixel
    for i=1:K
       res = regionprops(L{j},outMag(:,:,i),'MeanIntensity');
       gaborfeatures{j}(:,i) = [res.MeanIntensity]';
    end    
end
clearvars outMag K res

%Surf for training set
SurfFeatures = {datasetimages};
valid_points = {datasetimages};
all_points={datasetimages};
%Calculate surf features for each image
for i=1:datasetimages
    points = detectSURFFeatures(rgb2gray(Images{i}));
    [SurfFeatures{i}, valid_points{i}] = extractFeatures(rgb2gray(Images{i}),points);
    all_points{i}=uint8(valid_points{i}.Location);
end
 clearvars points

%Gray image

%SURF features for the gray image
 gray_points = detectSURFFeatures(grayImage);
 [gsurf, gsurf_points] = extractFeatures(grayImage,gray_points);
 
 clearvars gray_points


 %Superpixels for gray image
[Gray_L,Gray_Num] = superpixels(grayImage,superpixels_number,'Compactness',compact);

%Calculate Gabor features for gray image
outMag = imgaborfilt(grayImage,g);
K = size(outMag,3);
gray_gaborfeatures = zeros(Gray_Num,K+64);
for i=1:K
       res = regionprops(Gray_L,outMag(:,:,i),'MeanIntensity');
       gray_gaborfeatures(:,i) = [res.MeanIntensity]';
end


g_all_points=uint8(gsurf_points.Location);% all location of surf points
for j=1:size(g_all_points,1)
        sinx=g_all_points(j,1);
        siny=g_all_points(j,2);
         %populate each superpixel with Gabor and SURF features for gray image
        gray_gaborfeatures(Gray_L(sinx,siny),5:end)=gsurf(j,:);
end
clearvars sinx siny res


x = zeros(num_superpixels,68); %array of all superpixels in the training dataset
meangeneralvalues={1,datasetimages};
iterator = 0; %specifies the row on which the previous loop was

%Populate x array with Gabor features for SVM model
for i=1:datasetimages
    for j=1:size(gaborfeatures{i},1)
        for k=1:4
            x(iterator+j,k) = gaborfeatures{i}(j,k);
        end
    end
    iterator = iterator + size(gaborfeatures{i},1);
end

iterator =1;
%Populate x array with SURF features for SVM model
for i=1:datasetimages
    mean_surf=cell(1,NumLabels{i});
    for j=1:size(all_points{i},1)
        sinx=all_points{i}(j,1);
        siny=all_points{i}(j,2);
        mean_surf{1,L{i}(sinx,siny)}(end+1,:)=SurfFeatures{i}(j,:);  
    end
    for superpixel=1:NumLabels{i}
        if not(isempty(mean_surf{1,superpixel}))
        x(iterator,5:end)=mean(mean_surf{1,superpixel},1);
        end
        iterator=iterator+1;
    end
end

clearvars iterator sinx siny


fprintf("Creating model ...\n");

%model =loadLearnerForCoder('finalmodel'); %used to load a trained model
model=fitcecoc(x,y); % train model ( x = features for each superpixel, y = label of the class in which belongs)
%saveLearnerForCoder(model, 'finalmodel256'); % used to save a trained model

result = predict(model,gray_gaborfeatures); % predict using the model
imgrecreated = zeros(height,width,3);
%reconstruct image with predicted colors
fprintf("Coloring the image...\n");
for i=1:size(result)
    for j=1:height
        for k=1:width
            if isequal(Gray_L(j,k), i)
                       
                       imgrecreated(j,k,2:3)= centers(str2double(result(i)),1:2);
           end
            
        end
    end
end


labimg = rgb2lab( repmat(grayImage, [1 1 3]) ); %convert gray image to Lab
imgrecreated(:,:,1)=labimg(:,:,1);
a =lab2rgb(imgrecreated,'OutputType','uint8');
collage={datasetimages+2};
%Create a collage of training dataset with superpixels overlay
for i=1:datasetimages
    BW = boundarymask(L{i});
    collage(i)={imoverlay(Images{i},BW,'yellow')};
end
%Display the collage and the recolored image
BW = boundarymask(Gray_L);
collage(datasetimages+1)={imoverlay(grayImage,BW,'yellow')};
collage(datasetimages+2)={a};
montage(collage);
clearvars Gray_Num i index j k K L num_superpixels superpixel superpixels_number u x y 
figure (2), imshow(a); %colorized image


