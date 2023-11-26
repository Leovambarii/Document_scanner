# Document Photo Scanner

This Python project provides a document photo scanning application using OpenCV.
Functions together with interface were fully written using only `cv2` and `numpy` as a form of challange for an advanced computer graphics course project.

The application allows users to process an input image and perform various operations such as resizing, cropping, edge detection, and perspective transformation in order to achieve best results. Finally it allows saving of the results in form of component images or just the final processed image. The user can interact with the application through a graphical interface with trackbars for adjusting parameters and keyboard hotkeys for image saving and app closing.

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy

## Usage

To use the document photo scanner, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Leovambarii/Document_scanner.git
   cd Document_scanner
   ```

2. Install the required packages:

   ```bash
   pip install opencv-python numpy
   ```

3. Run the scanner:

   ```bash
   python paper_scan.py <img_path> <save_folder_path>
   ```

   - `<img_path>`: Path to the input image file.
   - `<save_folder_path>`: (Optional) Folder path where the results will be saved. If not provided, the default folder is used (`results`).

## Interface Controls

- **Keys:**
  - `q` or `Esc`: Close the application.
  - `w`: Save all components of the processed image.
  - `e`: Save the final processed image.
  - `r`: Save the final processed image in autobalanced black and white scale.

- **Trackbars:**
  - `Init cropping size`: Adjust the amount of pixels to be cropped from each side of the original scaled image initially.
  - `Edges additional contrast`: Control the additional contrast for edge detection.
  - `Edges threshold bottom`: Set the bottom threshold value for edge detection.
  - `Edges threshold top`: Set the top threshold value for edge detection.
  - `Cropping size`: Adjust the amount of pixels to be cropped from each side of the final image.
  - `Auto balance white&black`: Control the intensity of black and white autobalance.

## Results

- The application displays the processed component images with an interface for parameter adjustments.
- Users can save all components of the processed image and the final images based on their preferences.