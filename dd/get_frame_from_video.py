import cv2

import os
def get_frame_from_video(video_name, frame_time, img_dir, img_name):

    vidcap = cv2.VideoCapture(video_name)
    # Current position of the video file in milliseconds.
    vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time - 1)
    # read(): Grabs, decodes and returns the next video frame
    success, image = vidcap.read()

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if success:
        # save frame as JPEG file
        cv2.imwrite(img_dir + '/' + img_name, image)  

if __name__=='__main__':
    get_frame_from_video('s.flv',12,'output','we.jpg')