import cv2
import numpy as np

lk_params = dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
cap = cv2.VideoCapture('VID.mp4')
detector = cv2.QRCodeDetector()
ret, prev_frame = cap.read()
data, bbox, straight_qrcode = detector.detectAndDecode(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
dst_corners = bbox[0]

def calc_tilt(corners):
    top = np.linalg.norm(corners[0]-corners[1])
    right = np.linalg.norm(corners[1]-corners[2])
    bottom = np.linalg.norm(corners[2]-corners[3])
    left = np.linalg.norm(corners[3]-corners[0])
    tb_ratio = min(top,bottom)/max(top,bottom)
    lr_ratio = min(left,right)/max(left,right)
    tb_tilt = (1-tb_ratio)*90
    lr_tilt = (1-lr_ratio)*90
    return tb_tilt, lr_tilt

while True:
    colors = (255,255,255)
    ret, frame = cap.read()
    data, bbox, straight_qrcode = detector.detectAndDecode(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if data and bbox is not None:
        corners = bbox[0]
        colors = (255,0,0)
    else:
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, dst_corners,None, **lk_params)
        corners = p1
        colors=(0,0,255)
    if len(corners)==4:
        for i in range(len(corners)):
                pt1 = tuple(corners[i].astype(int))
                pt2 = tuple(corners[(i+1)% len(corners)].astype(int))
                cv2.line(frame, pt1, pt2, colors, 3)
    tb_tilt,lr_tilt = calc_tilt(corners)
    cv2.putText(frame, f'H_tilt = {tb_tilt:.1f}' , (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
    cv2.putText(frame, f'V_tilt = {lr_tilt:.1f}' , (10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
    cv2.imshow('QR Code', cv2.resize(frame,None,fx = 0.7,fy = 0.7))

    prev_frame = frame.copy()
    dst_corners = corners
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break