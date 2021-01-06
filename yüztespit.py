import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import locale
import RPi.GPIO as GPIO#GPIO erişimi sağlıyoruz

cam = cv2.VideoCapture(0)

Role = 14#14 nolu pin röle çıkışı olarak ayarlıyoruz
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)#pinlere erişim sağlıyor
GPIO.setup(Role, GPIO.OUT)
GPIO.output(Role,GPIO.LOW)#giriş te röleye 0 veriyor

zaman=time.ctime()#raspberry pi da kullanmak için 
locale.setlocale(locale.LC_TIME, "tr_TR") #raspberry pi da çalışmıyor zamanı işaretlemek için 

recognizer = cv2.face.LBPHFaceRecognizer_create()#ikili desenlere(siyah Beyaz) çevirerek haasrcascade anlıyacağı dile çeviriyor
recognizer.read('veri/veri.yml')#veri etiketleri kaydediyor
cascadePath = "haarcascade_frontalface_default.xml" #haarcascade veriler alınıyor
faceCascade = cv2.CascadeClassifier(cascadePath);#cascade verileri ile sınıflandırılıyor

isim = ['Bilinmiyor','samed','2.kişi','3.kişi']#sıreayla kaydedilen kişilerin isimleri yazılıyor 

while True:
    ret, img = cam.read()

    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yuzler = faceCascade.detectMultiScale(gri, 1.2,5)

    for (x, y, w, h) in yuzler:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)#dikdörtgen çiziliyor
        id, uyum = recognizer.predict(gri[y:y + h, x:x + w])

        if (uyum < 100):#eğer uyumlu ise alttaki işlemleri yap
            id = isim[id]
            uyum = f"{round(uyum,0)}%"
            GPIO.output(Role,GPIO.HIGH)#eğer doğru ise role yi çalıştır
            with open ("giris.txt","a",encoding="utf-8") as f:#kişi doğru ise giriş isimli txt dosyasına isim ve zaman bilgileri kayıt ediliyor
                f.write("\n" "ad="+id+" ==> giriş saati="+time.strftime("%a, %d %b %Y %H:%M:%S"))
                
        else:
            id = "bilinmiyor"
            uyum = "bilinmiyor"+f"=  {round(uyum,100)}%"
            GPIO.output(Role,GPIO.LOW)#eğer yanlış  ise role yi kapat

        cv2.putText(img, str(id), (x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 2)#Kişinin adı yazılıyor
 

    cv2.imshow('Guvenlik Kamerası', img)

    if cv2.waitKey(10)  == ord ('q') and cv2.waitKey(0):
        break               #döngüyü kırmamızı sağlıyor

cam.release()
cv2.destroyAllWindows()