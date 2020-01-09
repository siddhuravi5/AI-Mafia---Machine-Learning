import cv2
import numpy as np
import csv
frame=cv2.imread('C:\\Users\\Siddharth\\Pictures\\2018-02\\Before.jpg')
#frame=cv2.resize(frame,(500,800))
face_cascade=cv2.CascadeClassifier('C:\\Users\\Siddharth\\Documents\\haarcascade_frontalface_default.xml')
eyes_cascade=cv2.CascadeClassifier('C:\\Users\\Siddharth\\Documents\\haarcascade_eye.xml')
#eyes_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
f3=cv2.CascadeClassifier('C:\\Users\\Siddharth\\Documents\\haarcascade_mcs_nose.xml')
imgg=cv2.imread('C:\\Users\\Siddharth\\Pictures\\2018-02\\mustache2.jpg')
img2=cv2.imread('C:\\Users\\Siddharth\\Pictures\\2018-02\\glasses.jpg')
while True:
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    if(len(faces)==0):
       continue
    faces=sorted(faces,key=lambda x:x[2]*x[3])
    #print(faces)

    for face in faces[-1:]:
        x,y,w,h=face
        a=face
        #cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(10,15,205),-1)
    roi_gray = gray_frame[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
    eyes = eyes_cascade.detectMultiScale(gray_frame)
    print(eyes)
    for eye in eyes[:1]:
                ex,ey,ew,eh=eye
                #if(ex>20 and ey<600 and ex<350):
                #cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                X1=ex-18
                Y1=ey-11
                X2=X1+2*ew+55
                Y2=ey+eh+20
                img2=cv2.resize(img2,(X2-X1,Y2-Y1),interpolation=True)
                frame[Y1:Y2,X1:X2]=cv2.bitwise_and(frame[Y1:Y2,X1:X2],img2,mask=None)

    nose = f3.detectMultiScale(gray_frame)
    #print(eyes)
    for (ex,ey,ew,eh) in nose[:1]:
        
        #cv2.rectangle(gray_frame,(ex-15,ey+55),(ex+ew+15,ey+eh),(0,255,0),2)
        x1=ex-10
        y1=ey+30
        x2=ex+ew+25
        y2=ey+eh+20
        imgg=cv2.resize(imgg,(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)
        #frame[y1:y2,x1:x2]=cv2.addWeighted(frame[y1:y2,x1:x2],-5,imgg,-10,0,frame)
        frame[y1:y2,x1:x2]= cv2.bitwise_and(frame[y1:y2,x1:x2],imgg , mask=None)
    cv2.imshow("gray_frame",frame)
    frame=np.asarray(frame)
    frame=frame.reshape((-1,3))
    with open('C:\\Users\\Siddharth\\Documents\\'+'test_output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in frame:
            writer.writerow(i)
    #np.save('C:\\Users\\Siddharth\\Documents\\'+'test_output.csv',frame)
    break
    

print("data saved")
