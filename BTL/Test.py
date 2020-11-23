import numpy as np
import cv2
import pickle

########### PARAMETERS ##############
width = 640 # size khung hình
height = 480
threshold = 0.65 #MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0 # lap top  0; camera rời 1
#####################################

#### CREATE CAMERA OBJECT  - thông số camera
cap = cv2.VideoCapture(cameraNo) #cho phep truyc cap camera
cap.set(3,width)
cap.set(4,height)

#### LOAD THE TRAINNED MODEL
pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)

#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img) #cân bằng histogram
    img = img/255
    return img

# while True:
#     success, imgOriginal = cap.read()
#     img = np.asarray(imgOriginal)
#     img = cv2.resize(img,(32,32))
#     img = preProcessing(img)
#     cv2.imshow("Processsed Image",img)
#     img = img.reshape(1,32,32,1)
#     #### PREDICT
#     classIndex = int(model.predict_classes(img))
#     #print(classIndex)
#     predictions = model.predict(img)
#     #print(predictions)
#     probVal= np.amax(predictions)
#     print(classIndex,probVal)
#
#     if probVal> threshold:
#         cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
#                     (50,50),cv2.FONT_HERSHEY_COMPLEX,
#                     1,(0,0,255),1)
#
#     cv2.imshow("Original Image",imgOriginal)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#




img = cv2.imread('so3.jpg',1)
img = cv2.resize(img,(32,32))
img = preProcessing(img)

cv2.imshow("Processsed Image",img)
cv2.waitKey(0) # chờ ...

img = img.reshape(1,32,32,1)
    #### PREDICT
#classIndex = int(model.predict_classes(img))  #predict_classes: nhãn (cũ)
classIndex = int(model.predict_classes(img))  #predict_classes: nhãn
    #print(classIndex)
predictions = model.predict(img) # vecto tong xac xuat cua cac class
    #print(predictions)
probVal= np.max(predictions)
print('Number: {0}, persen: {1}'.format(classIndex,probVal))


# print('Chữ số: {0}, độ chính xác: {1} %'.format(classIndex,round(probVal*100, 2)))



