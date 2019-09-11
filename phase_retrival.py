import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase


path = 'D:\\1LAB\\phaseretrival\\'

sp = cv2.imread(path + 'SP.bmp',0)
bg = cv2.imread(path + 'BG.bmp',0)
row = sp.shape[0]
col = sp.shape[1]

def imgft(s):
    out = np.fft.fft2(s)
    shift = np.fft.fftshift(out)
    shift1 = shift.copy()
    shift1[(1*row)//4:row,0:col] = 0.01 #set a crop regeion approximatly to find max position
    place = np.where(shift1 == np.max(shift1))  #find max position cor
    newcenter = shift[int(place[0])-row//8:int(place[0])+row//8,int(place[1])-col//8:int(place[1])+col//8] #use max as center to crop a region
    inv_fshift = np.fft.ifftshift(newcenter)  # reverse fourier
    img_recon = np.arctan2(np.imag(np.fft.ifft2(inv_fshift)),np.real(np.fft.ifft2(inv_fshift)))   #calculate phase difference as wrapped image
    return img_recon


f_sp = imgft(sp)
f_bg = imgft(bg)
output1 = unwrap_phase(f_sp)   #unwrapped img
output2 = unwrap_phase(f_bg)

result = output1-output2   #divide two

plt.figure()
plt.imshow(result,cmap='jet')
plt.show()
