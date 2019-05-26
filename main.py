import cv2
import os
import time
import threading
import numpy as np
from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.properties import BooleanProperty, ObjectProperty, ListProperty, StringProperty
from kivy.clock import Clock
from kivy.uix.listview import ListItemButton
from PIL import Image as pilImage


FACEID = 0


class Detector(Image):
    def __init__(self, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.cv = cv2
        self.capture = self.cv.VideoCapture(self.cv.CAP_DSHOW)
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        self.minWidth = 0.1 * self.capture.get(3)
        self.minHeight = 0.1 * self.capture.get(4)
        self.font = self.cv.FONT_HERSHEY_SIMPLEX

        self.datasetPath = 'Dataset'
        self.trainerPath = 'Trainer/trainer.yml'

        faceCascadePath = 'Cascades/haarcascade_frontalface_default.xml'
        self.faceCascade = self.cv.CascadeClassifier(faceCascadePath)
        
        eyesCascadePath = 'Cascades/haarcascade_eye.xml'
        self.eyesCascade = self.cv.CascadeClassifier(eyesCascadePath)

        smileCascadePath = 'Cascades/haarcascade_smile.xml'
        self.smileCascade = self.cv.CascadeClassifier(smileCascadePath)

        self.recognizer = self.cv.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.trainerPath)

        Clock.schedule_interval(self.frames, 1.0 / 60)

        self.username = ''
        self.nameNow = 'None'
        self.confidenceNow = 0

        self.switchFaceDetector = False
        self.switchEyesDetector = False
        self.switchSmileDetector = False

        self.startRecognition = False
        self.startFaceTrainer = False

    def frames(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray = self.cv.cvtColor(frame, self.cv.COLOR_BGR2GRAY)
            if self.switchFaceDetector:
                self.faceDetector(frame, gray)

            buffer = self.cv.flip(frame, 0).tostring()
            imageTexture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
            imageTexture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = imageTexture

    def faceDetector(self, frame, gray):
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 17,
            minSize = (int(self.minWidth), int(self.minHeight)),
            flags = self.cv.CASCADE_DO_CANNY_PRUNING
        )

        for (x, y, w, h) in faces:
            self.cv.rectangle(frame, (x, y), (x+w, y+h), (166, 41, 211), 2)
            roiGray = gray[y: y+h, x: x+w]
            roiColor = frame[y: y+h, x: x+w]

            try:
                id, confidence = self.recognizer.predict(gray[y: y+h, x: x+w])

                if confidence < 100:
                    id = self.names[id]
                    self.nameNow = id
                    confidence = f'{round(100 - confidence)}%'
                    self.confidenceNow = confidence
                else:
                    id = 'unknown'
                    self.nameNow = id
                    confidence = f'{round(100 - confidence)}%'
                    self.confidenceNow = confidence
                
                self.cv.putText(frame, str(id), (x+5, y-5), self.font, 1, (255, 255, 255), 2)
                self.cv.putText(frame, str(confidence), (x+5, y+h-5), self.font, 1, (255, 255, 0), 1)
            except:
                pass

            if self.switchFaceDetector and self.switchEyesDetector:
                self.eyesDetector(roiGray, roiColor)

            if self.switchFaceDetector and self.switchSmileDetector:
                self.smileDetector(roiGray, roiColor)

            if self.startRecognition:
                self.faceDataset(gray, x, y, w, h)
                self.startRecognition = False

    def eyesDetector(self, roiGray, roiColor):
        eyes = self.eyesCascade.detectMultiScale(roiGray)

        for (x, y, w, h) in eyes:
            self.cv.rectangle(roiColor, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def smileDetector(self, roiGray, roiColor):
        smile = self.smileCascade.detectMultiScale(roiGray)

        for (x, y, w, h) in smile:
            self.cv.rectangle(roiColor, (x, y), ((x+w), (y+h)), (255, 0, 0), 2)

    def faceDataset(self, gray, x, y, w, h):
        global FACEID
        print('[INFO   ] Face Dataset started ...')
        count = 0
        while count <= 40:
            self.cv.imwrite(f'{self.datasetPath}/{FACEID}.{self.username}.{count}.jpg', gray[y: y+h, x: x+w])
            count += 1
        FACEID += 1
        print('[INFO   ] Face Dataset stoped ...')

    def getIdAndImages(self):
        print('[INFO   ] Get Id and Images started ...')
        imagePaths = [os.path.join(self.datasetPath, i) for i in os.listdir(self.datasetPath)]
        faceSamples = []
        ids = []
        names = []

        for imagePath in imagePaths:
            pilImg = pilImage.open(imagePath).convert('L')
            imgNumpy =  np.array(pilImg, 'uint8')
            splitImagePath = os.path.split(imagePath)[-1].split('.')
            imageID = int(splitImagePath[0])
            imageName = splitImagePath[1]

            faces = self.faceCascade.detectMultiScale(imgNumpy)

            for (x, y, w, h) in faces:
                faceSamples.append(imgNumpy[y: y+h, x: x+w])
                ids.append(imageID)
                names.append(imageName)
        print('[INFO   ] Get Id and Images stoped ...')
        return faceSamples, ids, names, imageID

    def faceTrainer(self):
        global FACEID
        print('[INFO   ] Training faces. It will take a few seconds. Wait ...')
        faces, ids, names, imageID = self.getIdAndImages()
        FACEID = imageID + 1
        self.names = self.uniqueSet(names)
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write(self.trainerPath)
        self.recognizer.read(self.trainerPath)
        print(f'[INFO   ] {len(np.unique(ids))} faces trained. Exiting Program')

    def uniqueSet(self, array):
        unique = [x for i, x in enumerate(array) if i == array.index(x)]
        return unique

    def exit(self):
        self.capture.release()
        self.cv.destroyAllWindows()


class LoginPopup(Popup):
    
    isAdmin = BooleanProperty(False)
    
    def login(self, username, password):
        if username == 'admin' and password == 'admin':
            self.isAdmin = True


class LoadingPopup(Popup):

    def __init__(self, **kwargs):
        super(LoadingPopup, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, .00000005)

    def update(self, dt):
        if self.ids._loadingProgressBar.value < 100:
            self.ids._loadingProgressBar.value += 1
        else:
            self.dismiss()


class FaceTrainPopup(Popup):
    
    def __init__(self, **kwargs):
        super(FaceTrainPopup, self).__init__(**kwargs)
        Clock.schedule_once(self.update)

    def update(self, dt):
        time.sleep(5)
        self.dismiss()


class StartScreen(Screen, AnchorLayout):
    
    def loading(self):
        loadingPop = LoadingPopup()
        loadingPop.open()


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.detector = Detector()
        Clock.schedule_interval(self.update, 1.0 / 20)

    def update(self, dt):
        self.ids._nameNow.text = self.detector.nameNow
        self.ids._confidenceNow.text = str(self.detector.confidenceNow)

    def playSimpleVideo(self):
        self.ids._mainDetector.add_widget(self.detector)
    
    def playFaceDetector(self):
        self.detector.switchFaceDetector = not self.detector.switchFaceDetector

    def playEyesDetector(self):
        self.detector.switchEyesDetector = not self.detector.switchEyesDetector

    def playSmileDetector(self):
        self.detector.switchSmileDetector = not self.detector.switchSmileDetector

    def openLoginPopup(self):
        lPop = LoginPopup()
        lPop.open()

    def exit(self):
        self.detector.exit()


class AdminScreen(Screen):

    imageNameList = ListProperty([])

    def __init__(self, **kwargs):
        super(AdminScreen, self).__init__(**kwargs)
        self.detector = Detector()
        self.imageNameList = self.getImages()
        self.imageNameList = self.uniqueSet(self.imageNameList)

    def playRecVideo(self):
        self.detector.switchFaceDetector = True
        self.ids._adminDetector.add_widget(self.detector)

    def recognitionFace(self):
        self.detector.username = self.ids._username.text
        self.ids._username.text = ''
        self.detector.startRecognition = True
        self.detector.startFaceTrainer = True
        faceTrainPop = FaceTrainPopup()
        faceTrainPop.open()

    def getImages(self):
        path = 'Dataset'
        print('[INFO   ] Get Images started ...')
        imagePaths = [os.path.join(path, i) for i in os.listdir(path)]
        names = []

        for imagePath in imagePaths:
            splitImagePath = os.path.split(imagePath)[-1].split('.')
            imageName = splitImagePath[1]
            names.append(imageName)

        names = self.uniqueSet(names)
        print('[INFO   ] Get Images stoped ...')
        return names

    def uniqueSet(self, array):
        unique = [x for i, x in enumerate(array) if i == array.index(x)]
        return unique
    
    def exit(self):
        self.detector.exit()


class Manager(ScreenManager):
    pass 


class OpencvApp(App):
    title = 'Face Detector'
    selectedValue = StringProperty(None)

    def build(self):
        screenManager = Manager()
        self.startScreen = StartScreen(name='startScreen')
        self.mainScreen = MainScreen(name='mainScreen')
        self.adminScreen = AdminScreen(name='adminScreen')
        screenManager.add_widget(self.startScreen)
        screenManager.add_widget(self.mainScreen)
        screenManager.add_widget(self.adminScreen)
        self.mainScreen.playSimpleVideo()
        self.adminScreen.playRecVideo()
        
        threading.Thread(target=self.mainScreen.detector.faceTrainer).start()

        return screenManager

    def on_start(self):
        self.startScreen.loading()
        Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        if self.adminScreen.detector.startFaceTrainer:
            self.adminScreen.detector.startFaceTrainer = False
            time.sleep(5)
            self.mainScreen.detector.faceTrainer()
            self.adminScreen.imageNameList = self.adminScreen.getImages()

    def select(self, value):
        self.selectedValue = value.text

    def deleteImage(self):
        path = 'Dataset'
        imagePaths = [os.path.join(path, i) for i in os.listdir(path)]
        names = []

        for imagePath in imagePaths:
            splitImagePath = os.path.split(imagePath)[-1].split('.')
            imageName = splitImagePath[1]
            if self.selectedValue == imageName:
                os.remove(imagePath)
            else:
                names.append(imageName)

        uniqueNames = self.uniqueSet(names)

        self.adminScreen.imageNameList = uniqueNames
        self.mainScreen.detector.faceTrainer()

    def uniqueSet(self, array):
        unique = [x for i, x in enumerate(array) if i == array.index(x)]
        return unique
   
    def on_stop(self):
        self.mainScreen.exit()
        self.adminScreen.exit()


if __name__ == "__main__":
    OpencvApp().run()
