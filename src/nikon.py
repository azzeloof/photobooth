import gphoto2 as gp
import sys
import os


class Nikon:
    def __init__(self):
        self.captureDir = "captures/nikon"
        try:
            self.camera = gp.Camera()
            self.camera.init()
            print("Camera initialized.")
        except gp.GPhoto2Error as ex:
            self.camera = None
            if ex.code == gp.GP_ERROR_MODEL_NOT_FOUND:
                print("Camera not found. Please make sure your camera is connected and supported.")
            print(f"An error occurred during initialization: {ex}")

    def capture(self):
        if self.camera != None:
            # Download the image
            try:
                file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
                target_path = os.path.join(os.getcwd(), self.captureDir, file_path.name)
                print(f'Downloading image to: {target_path}')
                camera_file = self.camera.file_get(
                    file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
                print("File retrieved from camera.")
                
                # Save file to computer
                camera_file.save(target_path)
                print("Image downloaded successfully.")
                return target_path
            except gp.GPhoto2Error as ex:
                print(f"An error occurred during download (segfault might happen here): {ex}")
                self.camera.exit()
                return None

    def release(self):
        if self.camera != None:
            self.camera.exit()


if __name__ == "__main__":
    c = Nikon()
    c.capture()
    c.release()