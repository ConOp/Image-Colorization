from skimage import io, color
LABImages=[]
for i in range(1,19):
    LABImages.append(color.rgb2lab(io.imread("Dataset/"+str(i)+".png")[:,:,:3])) #Read images then RGBA->RGB and RGB->LAB


'''rgba = io.imread("Dataset/1.png")
rgb=rgba[:,:,:3]
lab = color.rgb2lab(rgb)

io.imshow(rgb)
io.show()'''
