import sys
sys.path.append(r"D:\lab\CODE\phase retrival")
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import scipy
import scipy.stats
import time
import os
from scipy.ndimage import gaussian_filter1d,median_filter
import glob


def unwrapping(x):
    phase = x
    (ny, nx) = np.shape(phase)
    dx = np.zeros((ny, nx))
    dy = np.zeros((ny, nx))
    dx[:, :-1] = np.diff(phase, axis=1)
    dy[:-1, :] = np.diff(phase, axis=0)
    # dx[:, -1] = -phase[:, -1]
    # dy[-1, :] = -phase[-1, :]

    dxp = dx - 2 * np.pi * np.sign(dx) * np.floor(np.absolute(np.sign(dx) * np.pi + dx) / (2 * np.pi))
    dyp = dy - 2 * np.pi * np.sign(dy) * np.floor(np.absolute(np.sign(dy) * np.pi + dy) / (2 * np.pi))

    sum_der = np.zeros((ny, nx), dtype=np.float32)

    tmp = np.zeros((ny, nx), dtype=np.float32)
    for j in range(0, ny):
        for i in range(0, nx):
            c1 = 0 if i is nx - 1 else dxp[j, i]
            c2 = 0 if i is 0 else dxp[j, i - 1]
            c3 = 0 if j is ny - 1 else dyp[j, i]
            c4 = 0 if j is 0 else dyp[j - 1, i]
            sum_der[j, i] = (c1 - c2) + (c3 - c4)

            t = 2 * np.cos((np.pi * 4.0 * i) / (4.0 * nx)) + 2 * np.cos((np.pi * 4.0 * j) / (4.0 * ny)) - 4
            tmp[j, i] = t

    #     Forward DCT
    dst = scipy.fftpack.dct(scipy.fftpack.dct(sum_der, axis=1, norm="ortho").T, axis=1, norm="ortho").T

    #     (3) Solve Possion problem
    phi = np.true_divide(dst, tmp, out=np.zeros_like(dst), where=tmp != 0)
    # plt.imshow(phi)
    # plt.show()
    #     Inverse DCT
    phi = scipy.fftpack.idct(scipy.fftpack.idct(phi, axis=1, norm="ortho").T, axis=1, norm="ortho").T

    return phi

def imgft(s,i,j,bg_or_sp):     # x = 0 for bg x = 1 for sp
    out = np.fft.fft2(s)
    shift = np.fft.fftshift(out)
    shift1 = shift.copy()

    shift1[(1*row)*2//5:row,0:col] = 0.01 #set a crop regeion approximatly to find max position

    if bg_or_sp == "bg":
        place = np.where(shift1 == np.max(shift1))  #find max position cor
        print(place)
        newcenter = shift[int(place[0])-row//8:int(place[0])+row//8,int(place[1])-col//8:int(place[1])+col//8] #use max as center to crop a region
        center_pos = [place[0], place[1]]

    elif bg_or_sp == "sp":
        newcenter = shift[int(i) - row // 8 :int(i) + row // 8 ,int(j) - col // 8 :int(j) + col // 8 ]
        center_pos = [0,0]

    inv_fshift = newcenter  # reverse fourier
    # inv_fshift = np.pad(newcenter, ((384,384),(384,384)), 'constant',constant_values=0.1)  # reverse fourier
    
    img_amp = np.fft.ifft2(inv_fshift)**2  #calculate amplitude
    img_recon = np.arctan2(np.imag(np.fft.ifft2(inv_fshift)),np.real(np.fft.ifft2(inv_fshift)))   #calculate phase difference as wrapped image
    return img_recon,img_amp, center_pos,np.fft.ifft2(inv_fshift)



for i in range(1,10,1):

    print(i)
    t1 = time.time()
  
    sppath = r"D:\data\2020-08-27\Bead\100X\6\SP"+"\\"+"Buffer"+str(i)+".bmp"
    sp = cv2.imread(sppath,0)

    bgpath =r"D:\data\2020-08-27\Bead\100X\6\SP\Buffer90.bmp"
    bg = cv2.imread(bgpath,0)

    row = sp.shape[0]
    col = sp.shape[1]

    f_bg, a_bg, p1 ,new_bg= imgft(bg,0,0,"bg")    #find bg center first then directory used as sp img crop center after FT
    f_sp, a_sp, p2 , new_sp = imgft(sp,p1[0],p1[1],"sp")    #find bg center first then directory used as sp img crop center after FT

    f_result= -f_sp+f_bg
    a_result = np.sqrt(a_sp/a_bg)  # calculate amplitude
    output2 = unwrapping(f_result) #unwrapped img
    comb = output2+1j*(a_result.real)
    plt.imshow(comb.real ,cmap='jet',vmin = 0 , vmax = 2.8)

plt.show()

