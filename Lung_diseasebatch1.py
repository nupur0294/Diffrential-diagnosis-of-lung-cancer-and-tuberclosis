import time
#ss = time.time()
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import skimage.io
#from keras.backend import clear_session
from time import time
#from json import dumps
import os
import tensorflow
print(tensorflow.__version__)
from numpy import asarray, expand_dims, squeeze, int32, fromstring, uint8, ceil
from tensorflow import Session, Graph, import_graph_def, gfile, GraphDef
from cv2 import cvtColor, rectangle, putText, namedWindow, setWindowProperty, imshow,imwrite, waitKey, \
COLOR_RGB2BGR, FONT_HERSHEY_SIMPLEX, WND_PROP_FULLSCREEN, IMREAD_COLOR, imdecode, resizeWindow, \
destroyAllWindows, VideoCapture, VideoWriter, VideoWriter_fourcc,resize, FILLED, getTextSize
import cv2 
import pickle
import copy
import keras
import cv2
from keras.models import load_model, Model


#import time



import numpy as np


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library






class lung_class:

    def __init__(self, threshold):
        """
        This function initializes the class variables
        Arguments:
            ipaddress: This input is the IP address of gateway to send message
            port: This input is the port number of gateway to send message
            threshold: This input is the detection score threshold
        """
        #Bounding Box size to extend
        self.extend_ratio_w = 0.1
        self.extend_ratio_h = 0.1
        #Minimum threshold probability of a object to be detected
        self.score_threshold = threshold

        self.cnt = 0
        self.hand_list= {"1":"enlarged_lymph_node","2":"tree_bud","3":"TB_cavity", "4": "miliary_disease" ,"5": "cancer_nodule" , "6": "miliary_disease" , "7":"fibre"}

        self.frozen_graph = self.load_frozen_graph('/home/nupur/Desktop/lungcncer/cancer_Resnet27may/freeze/mdl/frozen_inference_graph.pb')

        self.image_tensor = self.frozen_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.frozen_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.frozen_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.frozen_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.frozen_graph.get_tensor_by_name('num_detections:0')
        self.sess = Session(graph=self.frozen_graph)
        self.Seatbelt_dict = { }

    def load_frozen_graph(self, model_path):
        '''
        Function to load the frozen protobuf file from the disk and parse it
        to retrieve the unserialized graph_def
        Arguments -
            model_path      : A string having the path of the tensorflow model(.pb).
        Returns -
            detection_graph : The unserialized graph_def that holds the network architecture.
        '''
        detection_graph = Graph()
        with detection_graph.as_default():
            od_graph_def = GraphDef()
            with gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                import_graph_def(od_graph_def, name='')
        return detection_graph

    def disease_detect(self, img ):
        """
        This function detects the object in the input frame
        Arguments:
            image_path: This input is the frame read from camera
        """
        # Read & convert jpeg image into numpy array
        left, right, top, bottom = (0,0,0,0)
        self.hand_list= {"1":"enlarged_lymph_node","2":"tree_bud","3":"TB_cavity", "4": "miliary_disease" ,"5": "cancer_nodule" , "6":        		"miliary_disease" , "7":"fibre"}
        font = FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 1
        font_color = (0,0,0)

        image_input_tmp = cvtColor(img, COLOR_RGB2BGR) # convert RGB to BGR
        image_np = asarray(image_input_tmp) # convert to numpy array

        im_height, im_width = image_np.shape[:2]
        image_np_expanded = expand_dims(image_np, axis=0)

        box, scos, clas, det = self.sess.run([self.boxes, self.scores, self.classes, \
self.num_detections], feed_dict={self.image_tensor: image_np_expanded})

        scos = squeeze(scos)
        box = squeeze(box)
        clas = squeeze(clas).astype(int32)
        det = squeeze(det)
        #val = 1 if len(box) >=2 else len(box)
        for num in range(0, len(box), 1):
            if scos[num] > self.score_threshold:
                (xmin, ymin, xmax, ymax) = (box[num][1], box[num][0], box[num][3], \
box[num][2])
        #print (xmin, ymin, xmax, ymax)
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin \
* im_height, ymax * im_height)
        #print (left, right, top, bottom)
        

        #object_image = copy.deepcopy(img[int(top):int(bottom),int(left):int(right)])
                new_top = top-((bottom-top)*self.extend_ratio_h)
                new_bottom = bottom+((bottom-top)*self.extend_ratio_h)
                new_left = left-((right-left)*self.extend_ratio_w)
                new_right = right+((right-left)*self.extend_ratio_w)

                if new_top < 0:
                    new_top = 0
                if new_bottom > im_height:
                    new_bottom = im_height
                if new_left < 0:
                    new_left = 0
                if new_right > im_width:
                    new_right = im_width
                #print (clas, "num", num, str(clas[0]))
       

                if str(clas[num]) in ['2' ,'3','1']: 
                    #print("match")
                    rectangle(img, (int(new_left), int(new_top)), (int(new_right), \
    int(new_bottom)), (0, 0, 255), 3)
                    print("1st", self.hand_list)
                    putText(img, self.hand_list[str(clas[num])], (int(left), int(top)), \
    FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    rectangle(img, (int(new_left), int(new_top)), (int(new_right), \
    int(new_bottom)), (0, 0, 255), 3)
                    #print(self.hand_list)
                    putText(img, self.hand_list[str(clas[num])], (int(left), int(top)), \
    FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print (self.hand_list[str(clas[num])])
		    
        return left, right, top, bottom
#s3= time.time()
#print("function defination" , s3-s2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python finding_CDR_ratio.py <image_name>")
        exit()
    im_name = sys.argv[1]
    image_path = "/home/nupur/Desktop/lungcncer/op/" + im_name
    out_path = "/home/nupur/Desktop/lungcncer/out/"
    img = cv2.imread(image_path)

    OD = lung_class( 0.8)
    left, right, top, bottom = OD.disease_detect(img )
    print( int(top) , int(bottom) , int(left) , int(right))
    #object_image = copy.deepcopy(img[int(top):int(bottom),int(left):int(right)])
    #imwrite("/home/nupur/Desktop/lungcncer/out/"+ im_name,object_image)
    
    
    
    # plt.imshow(image1)
    # plt.axis("off")
    # plt.show()
    
    left, right, top, bottom = OD.disease_detect(img)
    imwrite(out_path+im_name,img)
    #object_image = copy.deepcopy(img[int(top):int(bottom),int(left):int(right)])
    
    plt.imshow(img)
    plt.axis("off")
    plt.show()


'''
TRAINED_MODEL_PATH ="/home/nupur/Music/Mask_RCNN-master/samples/balloon/colour_detection/glaucoma/model_updated/colour_model.h5"
model = load_model(TRAINED_MODEL_PATH)
print ("here")
file_path = "/home/nupur/Music/Mask_RCNN-master/samples/balloon/out_hem/1.jpg"
img = cv2.imread(file_path )
imgs = cv2.resize(img, (120,120))
img1 = imgs.reshape(1,120,120,3)
predicted_classes = []
predicted_probs = []
classes = ['red','yellow',"white"] 
num_classes = len(classes)
#print (classes)
probabilities=model.predict(img1)
sorted_prob_idxs = (-probabilities).argsort()[0]
predicted_prob = np.amax(probabilities)
predicted_probs.append(predicted_prob)
predicted_class = classes[sorted_prob_idxs[0]]
predicted_classes.append(predicted_class)
'''
    
