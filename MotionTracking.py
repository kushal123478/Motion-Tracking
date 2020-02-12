# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:15:14 2019

@author: KUSHAL
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

# Harris Corner Detector
class HarrisCornerDetector:
    def InputImage(self, img, thres = 117):
        """Takes input image for the class"""
        self.img = img
        self.Thres = thres
        
    def Gradients(self):
        """
        Takes: image
        Returns: gradients in X and Y direction 
        Ix = dI/dx(u,v)
        Iy = dI/dy(u,v)
        """
        self.img = np.float64(self.img)
        Ix = cv2.Sobel(self.img/ 255, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(self.img/ 255, cv2.CV_64F, 0, 1, ksize=3)
        return Ix, Iy
    
    def StructureMatrix(self, Ix, Iy):
        """
        Takes: Gradients in X and Y direction
        Returns: M =  [Ixx, Ixy
                       Iyx, Iyy]
        """
        Ixx = np.multiply(Ix, Ix)
        Ixy = np.multiply(Ix, Iy)
        Iyx = np.multiply(Iy, Ix)
        Iyy = np.multiply(Iy, Iy)
        M = [[Ixx, Ixy], [Iyx, Iyy]]
        return M
    
    def Gaussian(self, image_matrix, sigma):
        """
        Takes: A matrix of images
        Returns: Gaussian filtered results of all images in matrix
        """
        gauss_matrix = []
        for image_list in image_matrix:
            gauss_list = []
            for image in image_list:
                gauss_image = cv2.GaussianBlur(image, (5, 5), sigma , sigma)
                gauss_list.append(gauss_image)
            gauss_matrix.append(gauss_list)
        return gauss_matrix
    
    def CorenerStrength(self, gauss_matrix):
        """
        Takes: Gaussian filtered Structure matrix
        Returns: Corner strngth image
        """
        det_M = np.multiply(gauss_matrix[0][0], gauss_matrix[1][1]) - np.multiply(gauss_matrix[0][1], gauss_matrix[1][0])
        trace = gauss_matrix[0][0] + gauss_matrix[1][1]
        alpha_trace = 0.06 * np.multiply(trace, trace)
        Q_uv = det_M - alpha_trace
        return Q_uv
    
    def ReturnCorners(self, Q_uv, image):
        """
        Takes: Corner strength function
        Returns: Locations of corners
        """
        corner1 = (Q_uv.copy() - np.min(Q_uv)) * 255 / (np.max(Q_uv) - np.min(Q_uv))
        corner = corner1.copy()
        corner1[corner1 > self.Thres] = 255
        corner1[corner1 <= self.Thres] = 0
        corner = corner * corner1
        GREEN = (0, 255, 0)
        corners = []
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        for x in range(2, image.shape[1] - 2):
            for y in range(2, image.shape[0] - 2):
                window = corner[y-2:y+3, x-2:x+3]
                if corner[y,x] != np.max(np.max(window)):
                    corner[y,x] = 0
        for x in range(0, image.shape[1]):
            for y in range(0, image.shape[0]):
                if corner[y,x] !=0:
                    cv2.circle(img, (x, y), 3, GREEN)
                    img[y,x] =  np.array([0, 255, 0])
                    corners.append((y,x))
        corners = np.asarray(corners)
        return img, corners
    
    def Detect(self):
        """
        Detects corners and saves the highlighted images as ab.png
        """
        Ix, Iy = self.Gradients()
        M = self.StructureMatrix(Ix, Iy)
        M_bar = self.Gaussian(M, 3)
        Q_uv = self.CorenerStrength(M_bar)
        Result, Corners = self.ReturnCorners(Q_uv, img)
        Result = np.uint8(Result)
        cv2.imwrite('CornerDetection.png', Result)
        return Result, Corners
       
if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\hotelImages\hotel.seq0.png", cv2.IMREAD_GRAYSCALE)
    HD = HarrisCornerDetector()
    HD.InputImage(img, thres = 89)
    Result, Corners = HD.Detect()
    #print(Corners)
    w = 7
    image_list = []
    for image_ind in range(50):
        image_list.append(cv2.imread(r"C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 2\hotelImages\hotel.seq"+str(image_ind) + ".png", cv2.IMREAD_GRAYSCALE))
    image_list = np.asarray(image_list)
    del(image_ind)
    
    Corners1 = np.array([])
    for i in range(Corners.shape[0]):
        x_i = Corners[i, 0]
        y_i = Corners[i, 1]
        if x_i - w >= 0 and y_i - w >= 0 and x_i +w <= img.shape[0] and y_i+w <= img.shape[1]:
            Corners1 = np.append(Corners1, Corners[i])
    Corners1 = Corners1.reshape(int(Corners1.shape[0] / 2), 2)
    
    def interpolated_window(image, pixel_loction, window_size):
        window = np.zeros((window_size, window_size))
        for i in range(window.shape[0]):
            window[i] = window[i] + np.arange(window_size) - int(window_size/2)
        x_add = window
        y_add = np.flip(window.T)
        interpol_x = pixel_loction[0] + x_add
        interpol_y = pixel_loction[1] + y_add

        x1 = np.floor(interpol_x)
        x1 = x1.astype(int)
        y1 = np.floor(interpol_y)
        y1 = y1.astype(int)
        x2 = np.ceil(interpol_x)
        x2 = x2.astype(int)
        y2 = np.ceil(interpol_y)
        y2 = y2.astype(int)
        
        x = pixel_loction[0]
        y = pixel_loction[1]
        
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                if x1[i][j] != x2[i][j] and y1[i][j] != y2[i][j]:
                    if x1[i][j] < 480 and x2[i][j] < 480 and y1[i][j] < 512 and y2[i][j] < 512:
                        window[i][j] = (image[x1[i][j]][y1[i][j]] * (x2[i][j] - x) * (y2[i][j] - y) + 
                              image[x2[i][j]][y1[i][j]] * (x - x1[i][j]) * (y2[i][j] - y) + 
                              image[x1[i][j]][y2[i][j]] * (x2[i][j] - x) * (y - y1[i][j]) + 
                              image[x2[i][j]][y2[i][j]] * (x - x1[i][j]) * (y - y1[i][j]) 
                              ) / ((x2[i][j] - x1[i][j]) * (y2[i][j] - y1[i][j]))
                else:
                    if x1[i][j] < 480 and x2[i][j] < 480 and y1[i][j] < 512 and y2[i][j] < 512:
                        window[i][j] = image[x1[i][j]][y1[i][j]]
            
        return window
    
    
    def Tracker(I1g, I2g, window_size, tau=1):
 
        kernel_x = np.array([[-1., 1.], [-1., 1.]])
        kernel_y = np.array([[-1., -1.], [1., 1.]])
        kernel_t = np.array([[1., 1.], [1., 1.]])#*.25

        I1g = I1g / 255.
        I2g = I2g / 255.

        mode = 'same'
        fx = sg.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
        fy = sg.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
        ft = sg.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + sg.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
        u = np.zeros(Corners1.shape[0])
        v = np.zeros(Corners1.shape[0])


        for k in range(Corners1.shape[0]):
            i = Corners1[k,0]
            j = Corners1[k,1]
            Ix = interpolated_window(fx, (i,j), window_size).flatten()
            Iy = interpolated_window(fy, (i,j), window_size).flatten()
            It = interpolated_window(ft, (i,j), window_size).flatten()
            
            b = np.reshape(It, (It.shape[0],1))
            A = np.vstack((Ix, Iy)).T # get A here
            
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
                u[k]=nu[1]
                v[k]=nu[0]
     
        return np.asarray((u,v)).T
    
    u_v = Tracker(image_list[0], image_list[1], 15, tau=1e-5)
    x_y_corners = np.zeros((len(image_list), Corners1.shape[0], 2))
    x_y_corners[0, :, :] = Corners1
    x_y_second = x_y_corners[0, :, :] + u_v
    for i in range(1, len(image_list)):
        x_y_corners[i, :, :] = x_y_corners[i-1, :, :] - u_v
        u_v = Tracker(image_list[i], image_list[i-1], 15, tau= 1e-5)
    
    img_res = Result.copy()
    IMG = cv2.cvtColor(image_list[49] , cv2.COLOR_GRAY2BGR)
    
    for k in range(20):
        j = np.random.randint(0, 693)
        for i in range(x_y_corners.shape[0]):
            x = x_y_corners[i][j][0]
            y = x_y_corners[i][j][1]
            if x < 480 and y < 512:
                img_res[int(x), int(y)][2] = 255
                IMG[int(x), int(y)][2] = 255
                cv2.circle(IMG, (int(y), int(x)), 1, (0, 0, 255))
                cv2.circle(img_res, (int(y), int(x)), 1, (0, 0, 255))
                
        
    cv2.imwrite("Motion_Tracking1.png", img_res)
    cv2.imwrite("Test1.png", IMG)
    Image_second = Result.copy()
    #Image_second = cv2.cvtColor(Image_second , cv2.COLOR_GRAY2BGR)
    for i in range(x_y_second.shape[0]):
        x = x_y_second[i][0]
        y = x_y_second[i][1]
        cv2.circle(Image_second, (int(y), int(x)), 1, (0, 0, 255))
        
    cv2.imwrite("Motion_Tracking_second1.png", Image_second)