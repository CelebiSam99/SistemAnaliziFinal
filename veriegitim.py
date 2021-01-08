import cv2
import numpy as np
from PIL import Image #fotoğraf kütüphanesi 
import os

path = 'yuzveri' 
recognizer = cv2.face.LBPHFaceRecognizer_create()#ikili desenlere(siyah Beyaz) çevirerek haasrcascade anlıyacağı dile çeviriyor
yuztespit = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#haarcascade kullanılarak yüz belirleniyor

def etiketleme(path): #burda etiketleme yapılıyor 
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    ornekler=[]
    ids = []
    for imagePath in imagePaths:                    
        PIL_img = Image.open(imagePath).convert('L')#fotoğrafları siyah beyza döndürüyor
        img_numpy = np.array(PIL_img,'uint8')#değerler imgnumpy aktarıluyor ki ilerde bu fotoğraflar sayısal olarak işlem yapılsın 
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        yuzler = yuztespit.detectMultiScale(img_numpy)
        #
        for (x,y,w,h) in yuzler:
            ornekler.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return ornekler,ids
print ("\n Yüz Eğitimi başladı")
yuzler,ids = etiketleme(path)
recognizer.train(yuzler, np.array(ids))#yüzler eğitiliyor

recognizer.write('veri/veri.yml') #yüz bilgilerinin olduğu dosya kayıt ediliyor
print("\n yüz eğitildi.")
