from skimage import io, color
rgb = io.imread(filename)
lab = color.rgb2lab(rgb)
