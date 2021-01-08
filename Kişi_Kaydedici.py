import cv2

kamera = cv2.VideoCapture(0)#kamera yı seçiyoruz
kamera.set(3, 640) # video genişliğini belirle
kamera.set(4, 480) # video yüksekliğini belirle
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#haarcascade kullanılarak yüz belirleniyor

veri = 60 #çekeceğimiz fotoğraf(veri) adadeini yazıyoruz


face_id = 1 #veri kaçıncı kişiye ait 

say = 0

while(True):
    ret, img = kamera.read()
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#fotoğrafları siyah beyaza çeviriyor
    yuzler = face_detector.detectMultiScale(gri, 1.3, 5)

    for (x,y,w,h) in yuzler:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)#dikdörtgen çiziyor
        say += 1 #her seferinde 1 arttırıyor

        cv2.imwrite("yuzveri/" + str(face_id) + '.' + str(say) + ".jpg", gri[y:y+h,x:x+w])#çekilen fotoğraflar etiketleniyor
        cv2.imshow('yüz verisi', img)#kamerayı açıyor


    if cv2.waitKey(10)  == ord ('q') and cv2.waitKey(0):
        break     
    elif say >= veri: #ya yukarıda tanımladığımız gibi q ve say 60 eşit ve büyük ise kapan
        break

kamera.release()
cv2.destroyAllWindows()
