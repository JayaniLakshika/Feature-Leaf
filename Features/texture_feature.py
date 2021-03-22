#Texture features
textures = mt.features.haralick(gray_img1)
ht_mean = textures.mean(axis=0)
contrast = ht_mean[1]
correlation_texture = ht_mean[2]
inverse_diff_moments = ht_mean[4]
entropy = ht_mean[8]
