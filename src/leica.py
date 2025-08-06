import logging
import os
import sys
import time
import gphoto2 as gp

def main():
    logging.basicConfig(
        format='%(levelname)s: %(name)s: %(message)s', level=logging.INFO)

    try:
        camera = gp.Camera()
        camera.init()
        print("Camera initialized.")
    except gp.GPhoto2Error as ex:
        if ex.code == gp.GP_ERROR_MODEL_NOT_FOUND:
            print("Camera not found. Please make sure your camera is connected and supported.")
            return 1
        print(f"An error occurred during initialization: {ex}")
        return 1

    print('Capturing image...')
    try:
        file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
        print(f'Image captured and stored on camera at: {file_path.folder}/{file_path.name}')
    except gp.GPhoto2Error as ex:
        print(f"An error occurred during capture: {ex}")
        camera.exit()
        return 1

    print("Waiting 2 seconds for camera to finish writing to card...")
    time.sleep(2)

    # Download the image
    target_path = os.path.join(os.getcwd(), file_path.name)
    print(f'Downloading image to: {target_path}')
    try:
        # Get file from camera
        camera_file = camera.file_get(
            file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
        print("File retrieved from camera.")
        
        # Save file to computer
        camera_file.save(target_path)
        print("Image downloaded successfully.")
    except gp.GPhoto2Error as ex:
        print(f"An error occurred during download (segfault might happen here): {ex}")
        camera.exit()
        return 1

    # Release the camera
    camera.exit()
    print("Camera released.")
    return 0

if __name__ == "__main__":
    sys.exit(main())