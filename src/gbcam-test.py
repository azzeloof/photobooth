import cv2
import numpy as np

cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

def gb_crop(frame):
    start_x = 128
    start_y = 128
    fx = 128
    fy = 112
    scale = 8
    end_x = int(start_x + fx*scale)
    end_y = int(start_y + fy*scale)
    frame = frame[start_y:end_y, start_x:end_x]
    frame = cv2.resize(frame, (fx, fy), interpolation=cv2.INTER_AREA)
    return frame

def gb_quantize(frame):
    frame = np.where((frame >= 0) & (frame <= 64), 0, frame)
    frame = np.where((frame >= 65) & (frame <= 128), 96, frame)
    frame = np.where((frame >= 129) & (frame <= 192), 178, frame)
    frame = np.where((frame >= 193) & (frame <= 255), 255, frame)
    return frame

while True:
    ret, frame = cam.read()
    try:
        frame = gb_crop(frame)
        frame = gb_quantize(frame)
    except:
        print("frame processing error")

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('c'):
        cv2.imwrite('frame.png', frame)


# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()