#Color features
red_channel = Acutal_img1[:,:,0]
green_channel = Acutal_img1[:,:,1]
blue_channel = Acutal_img1[:,:,2]
blue_channel[blue_channel == 255] = 0
green_channel[green_channel == 255] = 0
red_channel[red_channel == 255] = 0
        
red_mean = np.mean(red_channel)
green_mean = np.mean(green_channel)
blue_mean = np.mean(blue_channel)
        
red_std = np.std(red_channel)
green_std = np.std(green_channel)
blue_std = np.std(blue_channel)
