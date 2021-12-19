import cv2
import numpy as np
import torch
import torchvision
import opencv_transforms.functional as OTF
from torchvision import datasets
from PIL import Image

def k_means_clustering_algorithms(image, numberofclust=9):

    image_size = image.shape
    small_image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    image_sampling = small_image.reshape((-1, 3))
    image_sampling = np.float32(image_sampling)
    CT_formula = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    FG = cv2.KMEANS_PP_CENTERS
    
    _, _, centers = cv2.kmeans(image_sampling, numberofclust, None, CT_formula, 10, FG)
    centers = np.uint8(centers)
    color_scheme = []
     
    for i in range(0, numberofclust):
        main_color = np.zeros(image_size, dtype='uint8')
        main_color[:,:,:] = centers[i]
        color_scheme.append(main_color)
    
    return color_scheme

class Read_Dataset_Pair_images(datasets.ImageFolder):   

    def __init__(self, root, transform, art_sketch_ntw, numberofclust):
        super(Read_Dataset_Pair_images, self).__init__(root, transform)
        self.numberofclust = numberofclust
        self.art_sketch_ntw = art_sketch_ntw
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __getitem__(self, index):
        directory_P, label = self.imgs[index]
        image = self.loader(directory_P)
        image = np.asarray(image)
        image = image[:, 0:512, :]
        image = self.transform(image)
        color_scheme = k_means_clustering_algorithms(image, numberofclust=self.numberofclust)
        image = self.make_tensor(image)
        
        with torch.no_grad():
            image_edge = self.art_sketch_ntw(image.unsqueeze(0).to(self.device)).squeeze().permute(1,2,0).cpu().numpy()
            image_edge = OTF.to_grayscale(image_edge, num_output_channels=3)
            image_edge = OTF.to_tensor(image_edge)
            
        for i in range(0, len(color_scheme)):
            color = color_scheme[i]
            color_scheme[i] = self.make_tensor(color)

        return image_edge, image, color_scheme
    
    def make_tensor(self, image):
        image = OTF.to_tensor(image)
        image = OTF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image
    
class Read_Dataset_image(datasets.ImageFolder):   
 
    def __init__(self, root, transform, art_sketch_ntw, numberofclust):
        super(Read_Dataset_image, self).__init__(root, transform)
        self.numberofclust = numberofclust
        self.art_sketch_ntw = art_sketch_ntw
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __getitem__(self, index):
        directory_P, label = self.imgs[index]
        image = self.loader(directory_P)
        image = np.asarray(image)
        image = self.transform(image)
        color_scheme = k_means_clustering_algorithms(image, numberofclust=self.numberofclust)
        image = self.make_tensor(image)
        
        with torch.no_grad():
            image_edge = self.art_sketch_ntw(image.unsqueeze(0).to(self.device)).squeeze().permute(1,2,0).cpu().numpy()
            image_edge = OTF.to_grayscale(image_edge, num_output_channels=3)
            image_edge = OTF.to_tensor(image_edge)
            
        for i in range(0, len(color_scheme)):
            color = color_scheme[i]
            color_scheme[i] = self.make_tensor(color)

        return image_edge, image, color_scheme
    
    def make_tensor(self, image):
        image = OTF.to_tensor(image)
        image = OTF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image