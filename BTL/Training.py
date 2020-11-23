import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

# – Tiền xử lý dữ liệu
# – Dựng model
# – Train và ước tính model

# đối số
path = 'myData'
testRatio = 0.2     # 0.2%  cho test
valRatio = 0.2      # 0.2%  cho test
imageDimensions = (32,32,3)
batchSizeVal = 50
epochsVal = 10
stepsPerEpochVal = 2000

#### nhập dữ liệu hình ảnh từ thư mục
count = 0
images = []  # danh sách chứa tất cả hình ảnh
classNo = []  # danh sách chứa tất cả các id lớp của hình ảnh
myList = os.listdir(path)
print("Tổng số lớp đc phát hiện:", len(myList))
noOfClasses = len(myList)
print("nhập các lớp .......")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)  # ảnh hiện tại
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)  # thêm ảnh đã resize vào cuối danh sách
        classNo.append(x)  # lớp "số" đã xử lý -> append nhãn vào mảng
    print(x, end=" ")
print(" ")
print("Total Images in Images L"
      "ist = ", len(images))
print("Total IDS in classNo List= ", len(classNo))

#### chuyển đổi thành ma trận
images = np.array(images)
classNo = np.array(classNo)


print(images.shape)
print(classNo.shape)
print('------------')
# #### phân tích dữ liệu -
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio) # testSize = testRatio = 0.2
    # train_test_split: return 4 giá trị
    # images, classNo: data/lable  -> train_test_split: chia ra
    # chia ra X_train,X_test: images
    # chia ra y_train,y_test: classNo

X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=valRatio) # < train_test_split -> 0.8 >
    # tuong tu:
    # => test: 0.2 - 2k ảnh
    # => val 0.2  cua 0.8 (8k)

print(X_train.shape) # kích thước - (6502, 32, 32, 3) 6502: số lượng/ 32, 32 kích thước, 3 kênh màu
print(X_test.shape)
print(X_validation.shape)

#### biểu đồ phân bố dữ liệu của các Class nhãn
# numOfSamples= []
# for x in range(0,noOfClasses):
#     #print(len(np.where(y_train==x)[0]))
#     numOfSamples.append(len(np.where(y_train==x)[0]))
# print("X axis (0-9) : {0}".format(numOfSamples) )
#
# plt.figure(figsize=(10,5))
# plt.bar(range(0,noOfClasses),numOfSamples)
# plt.title("No of Images for each Class")
# plt.xlabel("Class ID")
# plt.ylabel("Number of Images")
# plt.show()





#### chức năng đánh giá trước hình ảnh để đào tạo (to Gray 7, color 0-1)
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow("PreProcesssed",img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing,X_train)))  # tap train -> xử lý qua hàm preProcessing
X_test = np.array(list(map(preProcessing,X_test)))    # xử lý qua hàm preProcessing
X_validation= np.array(list(map(preProcessing,X_validation))) # xử lý qua hàm preProcessing

#### tái tạo lại hình ảnh, quy tắc reshape?
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

#### hàm sinh ảnh - sinh thêm ảnh dựa trên dữ liệu đã có
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train) # đưa tập train vào để sinh theo mẫu dataGen

#### ONE HOT ENCODING OF MATRICES -> (Nhãn) to vecto
y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)




#### CREATING THE MODEL
def myModel():
    noOfFilters = 60  #số lượng filter n^2 - kernel, quy tắc:
    sizeOfFilter1 = (5 , 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2,2) # kích thước trượt
    noOfNodes= 500 #n^2 - noOfNodes, quy tắc:

    model = Sequential()   # bắt đầu khởi tạo model
    # convolutional layer và pooling layer
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], # add 1 lớp tích chập 2D, # activation: hàm kích hoạt
                      imageDimensions[1],1), activation='relu')))   # đầu vào model, input shape: kích thước ảnh đầu vào,
                                                                    # relu : tuyến tính -> phi tuyến
                                                                    # activation: khử tuyến tính
                                                                    # sigmoi:
                                                                    # relu: die relu -> các W về 0
                                                                    # tanh:

    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))

    model.add(MaxPooling2D(pool_size=sizeOfPool)) # giảm đi

    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))

    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))

    model.add(MaxPooling2D(pool_size=sizeOfPool))

    # end: convolutional layer và pooling layer

    #model.add(Dropout(0.5)) # bỏ qua ngẫu nhiên 1 vài ...

    model.add(Flatten()) # đập bẹt ()

    model.add(Dense(noOfNodes,activation='relu')) # lớp đầu teien của mạng nơ ron, noOfNodes: 500
            # mạng nơ ron 500 node tryền vào

    model.add(Dropout(0.5))   # giảm 0.5 số lượng % trong model - tránh hiện tượng model phức tạp quá nhưng dữ liệu lại đơn giản quá

    #out put
    model.add(Dense(noOfClasses, activation='softmax')) # phân lớp, k dùng để train/ dùng để phân loại

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        # setup  Adam: lr=0.001
        # loss='categorical_crossentropy' -> sai so
        # metrics = ['accuracy'] đánh giá
    return model
model = myModel()
print(model.summary())

#param : số lượng trọng số qua từng layer : ( kích thước filter + số lượng filter ) x số lượng filter (60)



#
#### STARTING THE TRAINING PROCESS
history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)




#### PLOT THE RESULTS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0) #evaluate: , verbose:0 - không hiển thị điểm/...
print('Test loss = ',score[0])
print('Test Accuracy =', score[1])

#### SAVE THE TRAINED MODEL
pickle_out= open("venv/model_trained.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()


