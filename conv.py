import cv2
from cv2 import VideoWriter_fourcc

temp = cv2.imread('out_vod/frame_out0.jpg')

h, w, l = temp.shape
fourcc = VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('media/out_vod.avi', fourcc, 30.0, (w, h))

for i in range(136):
    img = cv2.imread('out_vod/frame_out' + str(i) + '.jpg')
    vid.write(img)
    print(str(i * 100.0 / 136) + "%")

cv2.destroyAllWindows()
vid.release()
