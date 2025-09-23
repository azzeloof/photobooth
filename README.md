# Raspberry Pi Photobooth

This project is a fully-featured, multi-camera photobooth running on a Raspberry Pi. It integrates a low-resolution Game Boy Camera for a retro feel and a high-resolution Nikon DSLR for quality shots. The system is controlled by arcade-style hardware buttons and provides a live preview with on-screen instructions.

### Features
* **Dual Camera System**: Captures photos simultaneously from a Game Boy Camera and a Nikon DSLR.
* **Live Preview**: Displays a live feed from the Game Boy camera with interactive overlays for countdowns and instructions.
* **Hardware Controls**: Managed by an Adafruit Seesaw board, allowing for arcade button input to start sessions.
* **Automated Sessions**: Each session automatically takes a set number of photos with a configurable delay in between.
* **AI Upscaling**: An included feature processes the Game Boy camera images to create a larger, stylized version with a decorative frame.
* **Print Layouts**: Automatically arranges the captured photos onto a 6"x8" layout, ready for printing on a DS40 dye-sub printer.

## How It Works

The application is built in Python and runs a central event loop that manages the photobooth's state (e.g., `IDLE`, `COUNTDOWN`, `CAPTURING`).

1.  A user presses the start button, which queues a `StartSessionEvent`.
2.  The system enters a `COUNTDOWN` state, displaying a countdown on the screen.
3.  When the countdown finishes, a `CapturePhotoEvent` is triggered. The `CameraManager` captures images from both the Game Boy and Nikon cameras in a separate thread to keep the UI responsive.
4.  The captured images are processed; the Nikon image is cropped to the Game Boy's aspect ratio, and the Game Boy image is used to generate the AI upscaled version.
5.  This process repeats for the configured number of photos per session.
6.  Once all photos are taken, the `layout_page` function arranges them into a composite image, which is then sent to the printer.

## Setup and Usage

### Hardware Prerequisites
* Raspberry Pi
* Nikon DSLR camera (supported by `gphoto2`)
* Game Boy Camera with a USB interface
* Adafruit Seesaw controller with connected arcade buttons and LEDs
* DNP DS40 Photo Printer

### Software Installation
1.  Clone the repository.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install the `gphoto2` command-line utility:
    ```bash
    sudo apt-get update
    sudo apt-get install gphoto2
    ```
4.  Ensure your user is part of the `lp` group to manage the printer:
    ```bash
    sudo usermod -a -G lp $USER
    ```

### Running the Photobooth
To start the application, navigate to the `src` directory and run the main script:
```bash
cd photobooth/src
python3 photobooth.py
```
The application will initialize the hardware and start in fullscreen mode, ready for the first user to press the button.
