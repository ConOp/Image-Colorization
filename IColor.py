from skimage import io, color
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import slic
import numpy as np
LABImages=[]
for i in range(1,18):
    LABImages.append(color.rgb2lab(io.imread("Dataset/"+str(i)+".png")[:,:,:3])) #Read images then RGBA->RGB and RGB->LAB

ABimages=[]
for i in range(0,len(LABImages)):
    ABimages.append(LABImages[i][:,:,1:]) #Get only a and b values
for i in range(0,len(ABimages)):
    ABimages[i].shape=(16384,2) # Reshape from 128x128x2 to 16384,2
ABimages=np.asarray(ABimages)
ABimages.shape=(278528,2) # Reshape from 17,16384,2 to 278528,2



'''kmean = KMeans(n_clusters = 4)
kmean.fit(ABimages)
ykmean = kmean.predict(ABimages)
plt.scatter(ABimages[:,1],ABimages[:,0],c=ykmean,marker='.')
plt.scatter(kmean.cluster_centers_[:,1],kmean.cluster_centers_[:,0],c='r',marker='+')'''

segments = []
for i in range(len(LABImages)):
    segments.append(slic(LABImages[i],n_segments=5 ,compactness=10))
print (segments[13].shape)
