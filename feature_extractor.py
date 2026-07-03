import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from scipy.stats import entropy


class FeatureExtractor:

    def __init__(self):
        self.resize = (256, 256)

    def preprocess(self, img):

        img = cv2.resize(img, self.resize)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, gray

    def laplacian_variance(self, gray):

        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def edge_density(self, gray):

        edges = cv2.Canny(gray,100,200)

        return np.sum(edges>0)/edges.size

    def brightness(self, gray):

        return np.mean(gray)

    def contrast(self, gray):

        return np.std(gray)

    def reflection_ratio(self, gray):

        return np.sum(gray>240)/gray.size

    def fft_features(self, gray):

        fft=np.fft.fft2(gray)

        fft=np.fft.fftshift(fft)

        mag=np.log(np.abs(fft)+1)

        return [
            np.mean(mag),
            np.std(mag),
            np.max(mag),
            np.min(mag)
        ]

    def lbp_features(self,gray):

        radius=2

        points=16

        lbp=local_binary_pattern(gray,points,radius,"uniform")

        hist,_=np.histogram(
            lbp.ravel(),
            bins=np.arange(0,points+3),
            range=(0,points+2),
            density=True
        )

        return hist.tolist()

    def hog_features(self,gray):

        feat=hog(
            gray,
            orientations=9,
            pixels_per_cell=(16,16),
            cells_per_block=(2,2),
            feature_vector=True
        )

        return [
            np.mean(feat),
            np.std(feat),
            np.max(feat),
            np.min(feat)
        ]

    def color_features(self,img):

        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        feat=[]

        for i in range(3):

            feat.append(np.mean(hsv[:,:,i]))

            feat.append(np.std(hsv[:,:,i]))

        return feat

    def entropy_feature(self,gray):

        hist=cv2.calcHist([gray],[0],None,[256],[0,256])

        hist=hist.ravel()

        hist=hist/np.sum(hist)

        return entropy(hist)

    def sobel_features(self,gray):

        sx=cv2.Sobel(gray,cv2.CV_64F,1,0)

        sy=cv2.Sobel(gray,cv2.CV_64F,0,1)

        mag=np.sqrt(sx**2+sy**2)

        return [

            np.mean(mag),

            np.std(mag),

            np.max(mag)

        ]

    def extract(self,image_path):

        img=cv2.imread(image_path)

        if img is None:

            raise Exception(f"Cannot read {image_path}")

        img,gray=self.preprocess(img)

        feature=[]

        feature.append(self.laplacian_variance(gray))

        feature.append(self.edge_density(gray))

        feature.append(self.brightness(gray))

        feature.append(self.contrast(gray))

        feature.append(self.reflection_ratio(gray))

        feature.extend(self.fft_features(gray))

        feature.extend(self.hog_features(gray))

        feature.extend(self.color_features(img))

        feature.extend(self.sobel_features(gray))

        feature.append(self.entropy_feature(gray))

        feature.extend(self.lbp_features(gray))

        return np.array(feature,dtype=np.float32)


extractor=FeatureExtractor()


def extract_features(path):

    return extractor.extract(path)