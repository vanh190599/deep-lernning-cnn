import cv2
import os
def importDAta(path, images, classNo):
    count = 0
    # images = []  # danh sách chứa tất cả hình ảnh
    # classNo = []  # danh sách chứa tất cả các id lớp của hình ảnh
    myList = os.listdir(path)
    print("Tổng số lớp đc phát hiện):", len(myList))
    noOfClasses = len(myList)
    print("nhập các lớp .......")
    for x in range(0, noOfClasses):
        myPicList = os.listdir(path + "/" + str(x))
        for y in myPicList:
            curImg = cv2.imread(path + "/" + str(x) + "/" + y)  # ảnh hiện tại
            curImg = cv2.resize(curImg, (32, 32))
            images.append(curImg)  # thêm ảnh đã resize vào cuối danh sách
            classNo.append(x)  # lớp "số" đã xử lý
        print(x, end=" ")
    print(" ")
    print("Total Images in Images List = ", len(images))
    print("Total IDS in classNo List= ", len(classNo))