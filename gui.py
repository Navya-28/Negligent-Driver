import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from kivy.core.window import Window
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import winsound
from driver_prediction import predict_result
import threading
from Sign import predict_sign

class CamApp(App):

    def sound(self):
        winsound.PlaySound('Beta.wav', winsound.SND_FILENAME)

    def build(self):
        self.background = Image(source="back.jpg", allow_stretch = True)
        Window.add_widget(self.background)
        self.title = "Negligent Driver"
        self.OUTPUT_VIDEO_FILE = "output_driver.avi"
        self.OUTPUT_VIDEO_FILE1 = "output_sign.avi"
        self.img1=Image()
        self.img2=Image()
        layout = BoxLayout(spacing=30)
        layout.add_widget(self.img1)
        layout.add_widget(self.img2)
        
        self.writer = None
        self.writer1 = None
        (self.W, self.H) = (None, None)
        (self.W1, self.H1) = (None,None)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.score=0
        self.s=0
        
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        
        (grabbed, frame) = self.capture.read()
        (grabbed1, frame1) = self.capture1.read()
        height,width = frame.shape[:2]

        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]
            (self.H1, self.W1) = frame1.shape[:2]

        output = frame.copy()
        output1 = frame1.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame, (64, 64))

        frame1 = cv2.resize(frame1, (30, 30))

        frame = np.expand_dims(frame,axis=0).astype('float32')/255 - 0.5
        frame1 = np.expand_dims(frame1,axis=0)

        label = predict_result(frame)
        sign = predict_sign(frame1)


        if label != 'SAFE_DRIVING':
            text = "Activity: {}".format(label)
            self.score += 1
            cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)

        else:
            self.score -= 1

        if(self.score<1):
            self.score=1

        cv2.putText(output,'Score:'+str(self.score),(100,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1,cv2.LINE_AA)

        if (self.score>100):
            t1 = threading.Thread(target=self.sound)
            t1.daemon = True
            self.s += 1
            if self.s%100==0:
                t1.start()
                self.s=0

        if self.writer is None:

            fourcc = cv2.VideoWriter_fourcc(*"MPEG")
            self.writer = cv2.VideoWriter(self.OUTPUT_VIDEO_FILE, fourcc, 30,(self.W, self.H), True)

        self.writer.write(output)
        
        if sign != 'No vehicles':
            text1 = "Sign: {}".format(sign)
            cv2.putText(output1, text1, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)

        if self.writer1 is None:

            fourcc1 = cv2.VideoWriter_fourcc(*"MPEG")
            self.writer1 = cv2.VideoWriter(self.OUTPUT_VIDEO_FILE1, fourcc1, 30,(self.W1, self.H1), True)

        self.writer1.write(output1)
        
        buf1 = cv2.flip(output, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(output.shape[1], output.shape[0]), colorfmt='bgr') 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = texture1
        buf2 = cv2.flip(output1, 0)
        buf3 = buf2.tostring()
        texture2 = Texture.create(size=(output1.shape[1], output1.shape[0]), colorfmt='bgr') 
        texture2.blit_buffer(buf3, colorfmt='bgr', bufferfmt='ubyte')
        self.img2.texture = texture2

        def on_stop(self):
            self.capture.release()
            self.capture1.release()
            self.writer.release()
            self.writer1.release()


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()