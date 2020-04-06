# import libraries of python OpenCV
import cv2
# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
# calling the cascadeclassifier passing the file location
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# First of all, we have to extract the fertures of the face...
# create a function in which we will pass an image.
def faceExtractor(img):
    # we will find image inside the RGB but we can't perform any operation on RGB image so need to
    # convert into Gray scale image...
    gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # face classifier ---> detectMultiScale(image, scaleFactor, minNeighbors)
    faces = face_classifier.detectMultiScale(gry,1.3,5)
    # this condition  check the here is not the value of face so return the None
    if faces is():
        return None
    # if value of face so we will pass the coordinator the x,y, w,h
    for (x,y,w,h) in faces:
        # whatever changes are being made, we will store in "cropped_faces" variable...
        # create a list start y to y+h it will go. and start x to x+w it will go...
        cropped_faces = img[y:y+h, x:x+w]
        return cropped_faces

# Here we will configure the camera function using VideoCapture function, zero is a bydefult camera id...
# capture frames from a camera
cap = cv2.VideoCapture(0)
# count is variable it will help the count the image
count = 0

while True:
    # reads frames from a camera
    ret, frame = cap.read()
    # if it will is not empty so we count the value in count variable
    if faceExtractor(frame) is not None:
        count+=1
        # will need to resizing the face because it will be size of the camera...
        # which the faceExtractor function in which pass the frame
        face = cv2.resize(faceExtractor(frame),(200,200))
        # we will the convert the resize face into gray...
        # convert to gray scale of each face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # store the value of face(images)
        fileNamePath = "F:/PycharmProjects/opencv/faceCollection/user" +str(count)+'.png'
        # write the faces into a folder using imwriter function.
        cv2.imwrite(fileNamePath,face)
        # count the value of face
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),2)
       # show the output croped face
        # Display an image in a window
        cv2.imshow("Face cropper",face)
    else:
        print("Face not found")
        pass
    # Wait for Esc key to stop
    if cv2.waitKey(1) ==13 or count==120:
        break
# Close the window
cv2.resize()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
print("Colleting samples completed !")