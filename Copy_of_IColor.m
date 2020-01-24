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

num_superpixels = 0;
for i=1:17    
    num_superpixels = num_superpixels + max(max(L{i}));
    clear max;
end

y = strings([num_superpixels,1]);
index = {17};
adder = 0;
for u=1:17
    psa = zeros(max(max(L{u})),64); %stiles 64 = classes apo kmeans | seires = pli8os superpixel
    for i=1:128
        for j=1:128
            sup = L{u}(i,j); %gia ka8e eikona, gia ka8e pixel se poio superpixel anoikei
            psa(sup,img{u}(i,j)) = psa(sup,img{u}(i,j)) + 1;
        end
    end
    [max,index{u}] = max(psa');
    clear max;
    for i=1:max(max(L{u}))
       y(i+adder,1) = ['Class_',int2str(index{u}(i))];
    end
    adder = adder + i;
    clear max;
end
