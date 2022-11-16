import cv2
import mediapipe as mp
import numpy as np
import time

wCam, hCam = 640,480 #menentukan resolusi kamera yang digunakan

cam = cv2.VideoCapture(0) #memanggil fungsi untuk membuka aplikasi kamera
cam.set(3,wCam)
cam.set(4,hCam)
ptime = 0 #menentukan previous waktu untuk mencari fps 
handNo= 0 #setelan awal untuk tangan adalah 0

# 0 = pangkal tangan
# 1 = tangan ibu jari
# 2 = pangkal ruas ibu jari pertama
# 3 = pangkal ruas ibu jari kedua
# 4 = ujung ibu jari
# 5 = pangkal ruas jari telunjuk pertama
# 6 = pangkal ruas jari telunjuk kedua
# 7 = pangkal ruas jari telunjuk ketiga
# 8 = ujung jari telunjuk
# 9 = pangkal ruas jari tengah pertama
# 10 = pangkal ruas jari tengah kedua
# 11 = pangkal ruas jari tengah ketiga
# 12 = ujung jari tengah
# 13 = pangkal ruas jari manis pertama
# 14 = pangkal ruas jari manis kedua
# 15 = pangkal ruas jari manis ketiga
# 16 = ujung jari manis
# 17 = pangkal ruas jari kelingking pertama
# 18 = pangkal ruas jari kelingking kedua
# 19 = pangkal ruas jari kelingking ketiga
# 20 = ujung jari kelingking

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cam.read() #membuat keadaan ketika success terpenuhi akan membaca fungsi cam

    imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results =hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: 
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)
            
            myHand = results.multi_hand_landmarks[handNo]
            lmList = []
            for titik, lm in enumerate(myHand.landmark):
                #print(titik,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(titik,cx,cy)
                lmList.append([titik, cx,cy])
                
            if len(lmList) !=0:
                print(lmList[1])
    

    #frame rate; ctime (waktu real); ptime(waktu terbarukan)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime


    #memasukkan fps pada jendela camera
    cv2.putText(img, f'FPS : {int (fps)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2) #(img, f'teks', (sb x, sb y), font, size, warna, thickness)

    cv2.imshow("Img",img) #menampilkan kamera pada tab windows dengan nama Img
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
# Destroy all the windows
cv2.destroyAllWindows()

