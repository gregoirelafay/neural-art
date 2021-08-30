import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_absolute_error
import cv2

import fourier

MAIN_PATH =  "../raw_data/wikiart/" # Path to the directory which contains CSVs and the folder 'dataset'
IMAGES = "dataset"
CSV_NAME = "wikiart-movement-genre_True-class_8-merge_mov-1-n_1000_max.csv" # Nwikiart-movement-genre_True-class_3-merge_test1-n_1000_max.csvme of the CSV we want to use
NUM_MOVEMENT = 8 # Number of movements to classify
NUM_GENRE = 10 # Number of genres to classify
IMG_HEIGHT = IMG_WIDTH = 224 # Model's inputs shapes

root=MAIN_PATH + IMAGES + '/'
X_BASE=pd.read_csv(root + CSV_NAME)


def baselines_viz_single(img,n_colors = 5,plot=True,array=True,rgb_fft='g'):

    '''
        Function to compute average color, n-dominant colors and Fourier
        magnitude spectrum from an image

        #requires opencv-python

        img : unflattened image to be studied - np.ndarray (ex shape : 603,325,3)
        n-colors : n-dominant colors to be displayed
        plot : True -> display plot
        array : True -> returns a dictionary of above computed values
        rgb_fft : [r,g,b] -> color for FFT display

    '''

    #resize and flatten
    img_224=np.array(tf.image.resize(img,[224,224]))
    img_pixel = np.float32(img_224.reshape(-1, 3))

    #average color
    avg_color=img_pixel.mean(axis=0)
    img_pixel.shape

    #KMean for cluster identification of dominant colors
    pixels=np.float32(img_pixel)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, .5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dom_color = palette[np.argmax(counts)]


    #Magnitude Spectrum

    rgb_dict={'r':0,'g':1,'b':2}

    U=np.array(img_224[:,:,rgb_dict[rgb_fft]],dtype=np.float64)
    V=np.fft.fft2(U)
    VC = np.fft.fftshift(V)
    P = np.power(np.abs(VC),2)
    img_tff = fourier.matriceImageLog(P,[1,0,0])


    #Viz functions
    avg_patch = np.ones(shape=(224,224,3), dtype=np.uint8)*np.uint8(avg_color)

    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    rows = np.int_(224*freqs)

    dom_patch = np.zeros(shape=(224,224,3), dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(16,10))
    ax0.imshow(img)
    ax0.set_title('Original Image')
    ax0.axis('off')
    ax1.imshow(np.array(img_224,dtype=int))
    ax1.set_title('Resized Image')
    ax1.axis('off')
    ax2.imshow(avg_patch)
    ax2.set_title('Average color')
    ax2.axis('off')
    ax3.imshow(dom_patch)
    ax3.set_title('Dominant colors')
    ax3.axis('off')
    ax4.imshow(img_tff)
    ax4.set_title('Magnitude Spectrum')
    ax4.axis('off')

    #output selection

    if plot :
        plt.show(fig)

    weights=counts/pixels.shape[0]
    out=None
    if array:

        img_dict={}
        img_dict['avg_color']=avg_color
        img_dict['dom_color']=palette
        img_dict['dom_weights']=weights
        img_dict['magnitude_spectrum']=img_tff

    return img_dict


def baselines_viz_mov(df,root,mov,n_colors = 5,plot=True,array=True,
                      max_iter=10000,epsilon=0.5,rgb_fft='g'):

    '''
        Function to return a dictionnary of computed average color, dominant
        n-colors clustered by Kmeans and mean FFT Magnitude Spectrum of a
        chosen painting movement


        df : DataFrame ; initial dataframe with filename and movement
        mov : String ; chosen movement for baseline observation
        plot : Bool ; chose to print the colors when executing the function
        array : Bool ; returns the arrays for colors : (avg_color,palette)
        max_iter : max iterations for Kmean clustering
        epsilon : Kmean convergence epsilon

    '''

    path_list=df['file_name'][df['movement']==mov]
    pixels=np.empty(3)
    fourier_img=np.zeros((224,224, 3))

    for file in list(path_list):
        #average and dominant colors :
        #images are reshaped to 224X224 and pixels are stacked next to each other
        path=root+file
        img=plt.imread(path)
        img_224=np.array(tf.image.resize(img,[224,224]))
        img_pixel = np.float32(img_224.reshape(-1, 3))
        pixels=np.vstack((pixels,img_pixel))


        #Magnitude Spectrum on selected color

        rgb_dict={'r':0,'g':1,'b':2}

        U=np.array(img_224[:,:,rgb_dict[rgb_fft]],dtype=np.float64)
        V=np.fft.fft2(U)
        VC = np.fft.fftshift(V)
        P = np.power(np.abs(VC),2)
        img_tff = fourier.matriceImageLog(P,[1,0,0])
        fourier_img += img_tff

    avg_color=pixels.mean(axis=0)


    #Kmean clusterization to identify
    #first pixel for initialization, to be removed
    pixels=np.float32(pixels)[1:]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dom_color = palette[np.argmax(counts)]

        ##Magnitude Spectrum

    magnitude_spectrum = fourier_img/len(path_list)

        #Viz functions
    avg_patch = np.ones(shape=(224,224,3), dtype=np.uint8)*np.uint8(avg_color)

    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    rows = np.int_(224*freqs)

    dom_patch = np.zeros(shape=(224,224,3), dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])


    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16,10))
    ax0.imshow(avg_patch)
    ax0.set_title(mov.capitalize()+' - Average color')
    ax0.axis('off')
    ax1.imshow(dom_patch)
    ax1.set_title(mov.capitalize()+' - Dominant colors')
    ax1.axis('off')
    ax2.imshow(magnitude_spectrum, cmap = 'gray')
    ax2.set_title('Magnitude Spectrum')
    ax2.axis('off')
    plt.show(fig)

    #output selection

    if plot :
        plt.show(fig)


    weights=counts/pixels.shape[0]
    out=None
    if array:
        out =(avg_color,palette,weights,magnitude_spectrum)

    return out


def base_pred_avg(img,base_dict):

    """
        Predict the movement of an image from the average color

        img : image for which to determine movement
        base_dict : dictionnary from a trained baseline instance

    """

    mov_list=list(base_dict.keys())

    loss_dict={}

    for mov in mov_list:
        loss_dict[mov]=mean_absolute_error(img, base_dict[mov]['avg_color'])

    return min(loss_dict, key=loss_dict.get)


def base_pred_dom(img,base_dict,n_colors):

    '''
        Description : movement prediction based on dominant colors for an image

        img : image for which to determine movement
        base_dict : dictionnary from a trained baseline instance

    '''

    #each color gets compared to dominant colors of every movement
    #rating is done using MAE weighted by the inverse of dominance weights
    #minimal weighted distance is then used as a loss

    mov_list=list(base_dict.keys())

    loss_dict={}

    for mov in mov_list:
        loss_mov=0
        for color_dom in list(range(0,n_colors)):
            loss_list=[]
            for color_mov in list(range(0,n_colors)):
                loss_list.append((base_dict[mov]['dom_weights'][color_mov]*img['dom_weights'][color_dom])**-1 *\
                    mean_absolute_error(img['dom_color'][color_dom]/255,base_dict[mov]['dom_color'][color_mov]/255))
            loss_mov+=min(loss_list)
        loss_dict[mov]=loss_mov

    return min(loss_dict, key=loss_dict.get)


def base_pred_fft(img,base_dict):

    '''
        Description : movement prediction based on FFT of an image

        img : image for which to determine movement
        base_dict : dictionnary from a trained baseline instance

    '''

    mov_list=list(base_dict.keys())

    loss_dict={}

    for mov in mov_list:
        loss_dict[mov]=5*np.abs(img['magnitude_spectrum']-base_dict[mov]['mov_fft']).sum()

    return min(loss_dict, key=loss_dict.get)


class Baseline(object):
    def __init__(self,X,y,X_BASE,n_colors=5):
        """
            X : panda DataFrame
            y : panda Series
            X_BASE : base dataframe of 1000 paintings per movement for color trainings

        """

        self.X=X
        self.y=y
        self.X_BASE=X_BASE
        self.n_colors = n_colors
        self.path=root


    def occurence(self):

        """
            Returns the prediction based on occurence of movements in the
            train dataset

        """

        baseline_mov=dict(self.X.groupby(by='movement')['file_name'].count()/self.X.shape[0])
        baseline_gen=dict(self.X.groupby(by='genre')['file_name'].count()/self.X.shape[0])

        return baseline_mov , baseline_gen


    def basedict(self):

        """
            Compute the baseline dictionnary summaryzing mean FFT, average and
            dominant colors for each movement

        """

        mov_list=list(self.X['movement'].unique())

        base_dict={}
        for mov in mov_list:
            mov_dict={}
            baseline=baselines_viz_mov(self.X,root,mov,self.n_colors,plot=False,array=True)
            mov_dict['avg_color']=baseline[0]
            mov_dict['dom_color']=baseline[1]
            mov_dict['dom_weights']=baseline[2]
            mov_dict['mov_fft']=baseline[3]
            base_dict[mov]=mov_dict

        return base_dict


    def prediction(self):

        """
            Compute the baseline dictionnary summaryzing accuracy and
            prediction score for each movement using all methods trained above

        """

        ####### TO BE DONE ######

        pass