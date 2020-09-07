import cv2
import dlib
import time
import numpy as np
t=time.time()

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

pixel_w = 0.000825
pixel_h = 0.000825
f = 0.35

face_w = 18
face_h = 18

gaze = [0, 0]

input_w = 320

def gaze2vec3d(input_angle):
    # change 2D angle to 3D vector
    # input_angle: (vertical, horizontal)
    vec = [np.sin(input_angle[1])*np.cos(input_angle[0]), 
                np.sin(input_angle[0]), 
                np.cos(input_angle[1])*np.cos(input_angle[0])]
    return vec

def compute_Rcf(x, y, z):
    sx = y / (y**2+z**2)**0.5 
    cx = z / (y**2+z**2)**0.5 
    sy = 0
    cy = 1
    sz = x / (x**2+y**2)**0.5
    cz = y / (x**2+y**2)**0.5

    R = np.zeros((4, 4))
    R[:, 3] = [x, y, z, 1]

    R[:3, :3] = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx], 
                            [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], 
                            [-sy, cy*sx, cy*cx]])

    return R



while(True):
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    img_h, img_w, _ = img.shape
    cent_h, cent_w = (img_h//2, img_w//2)
    input_h = int(1.0 * img_h / img_w * input_w)
    scale = 1.0 * img_w / input_w
    img_res = cv2.resize(img, (input_w, input_h))
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        left = int(face.left()*scale)
        top = int(face.top()*scale)
        right = int(face.right()*scale)
        bottom = int(face.bottom()*scale)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        point_w, point_h = ((left + right)//2, (top+bottom)//2)

        ratio = face_w / ((right - left) * pixel_w)
        
        x = face_w * (point_w - cent_w) / (right - left)
        y = face_h * (point_h - cent_h) / (bottom - top)
        z = f * ratio
        p0 = np.array([x, y, z])
        vec3d = gaze2vec3d(gaze)
        vec3d_e = np.hstack([vec3d, [1]]).reshape((4, -1))
        Rcf = compute_Rcf(x, y, z)
        print(Rcf)
        p1 = np.dot(Rcf, vec3d_e)[:3].reshape((-1))

        xg = p1[0] - p0[2] / (p1[2]-p0[2]) * (p1[0]-p0[0]) 
        yg = p1[1] - p0[2] / (p1[2]-p0[2]) * (p1[1]-p0[1]) 
        print(xg, yg)



        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, 'X:%0.2f, Y:%0.2f, Z:%0.2f' % (x, y, z), (cent_w-200, 50), font, 1.2, (0, 0, 255), 2)
        

    cv2.imshow("face", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()

