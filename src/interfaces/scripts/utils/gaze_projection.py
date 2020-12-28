import cv2
import dlib
import time
import numpy as np
t=time.time()


# fx = 633.06324599
# fy = 633.06324599
# u0 = 441.39804136
# v0 = 257.74826685

fx = 1536
fy = 1536
u0 = 960
v0 = 540

cent_h, cent_w = (int(v0), int(u0))

face_w = 16
face_h = 16



input_w = 320

def gaze2vec3d(input_angle):
    # change 2D angle to 3D vector
    # input_angle: (vertical, horizontal)
    vec = [np.sin(input_angle[1])*np.cos(input_angle[0]), 
                np.sin(input_angle[0]), 
                np.cos(input_angle[1])*np.cos(input_angle[0])]
    return vec

def compute_Rcf(x, y, z):
    esp = 0.000001
    sx = y / ((z**2+y**2)**0.5+esp)
    cx = -z / ((z**2+y**2)**0.5+esp)
    sy = x / ((z**2+x**2)**0.5+esp)
    cy = z / ((z**2+x**2)**0.5+esp)
    sz = 0
    cz = 1

    R = np.zeros((4, 4))
    R[:, 3] = [x, y, z, 1]

    R[:3, :3] = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx], 
                            [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], 
                            [-sy, cy*sx, cy*cx]])

    return R




def gaze_to_screen(gaze, face, scale=0.25):
    left = int(face.left() / scale)
    top = int(face.top() / scale)
    right = int(face.right() / scale)
    bottom = int(face.bottom() / scale)

    point_w, point_h = ((left + right)//2, (top+bottom)//2)
    
    x = 1. * face_w * (point_w - cent_w) / (right - left)
    y = 1. * face_h * (point_h - cent_h) / (bottom - top)
    z = 1. * face_w / (right - left) * fx


    p0 = np.array([x, y, z])
    vec3d = gaze2vec3d(gaze)
    vec3d_e = np.hstack([vec3d, [1]]).reshape((4, -1))
    Rcf = compute_Rcf(x, y, z)

    p1 = np.dot(Rcf, vec3d_e)[:3].reshape((-1))

    xg = p0[0] - p0[2] / (p1[2]-p0[2]) * (p1[0]-p0[0]) + 2
    yg = p0[1] - p0[2] / (p1[2]-p0[2]) * (p1[1]-p0[1]) 

    xg = -xg
    gaze_p = np.array([xg, yg])
    face_p = p0

    return gaze_p, face_p



if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)

    scale = 0.25
    gaze = [0, 0]

    while(True):
        ret, frame = cap.read()

        frame = frame[:,::-1,:].copy()
        frame = cv2.resize(frame, (1920, 1080))
        frame_small = cv2.resize(frame, None, fx=scale, fy=scale,interpolation = cv2.INTER_CUBIC)
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects_small = detector(gray_small, 1)

        for face in rects_small:
            gaze_p, p0 = gaze_to_screen(gaze, face, scale)
            print(gaze_p)
            left = int(face.left() /scale)
            top = int(face.top() /scale)
            right = int(face.right()/scale)
            bottom = int(face.bottom() /scale)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(frame, 'X:%0.2f, Y:%0.2f, Z:%0.2f' % (p0[0], p0[1], p0[2]), (cent_w-200, 50), font, 1.2, (0, 0, 255), 2)
            

        cv2.imshow("face", frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()