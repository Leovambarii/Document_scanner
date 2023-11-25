import cv2
import sys
import os
import numpy as np

#TODO add sliders
#TODO add edition loop
#TODO tests on multiple different images
#TODO add img labels on display?
#TODO sliders for blur, edges, crop size, autobalance
FOLDER_SAVE = "results"
MAX_LEN_IMG = 2048 # Maximum width or height that orginal image will be proportionally resized
MIN_CONTOUR_AREA_PERC = 30 # Minimum percentage of image that a best countour should have (int in range 0 to 100)
CROP_SIZE = 25 # Amount of pixels that will be cropped from each side of image for final image
KEY_CLOSERS = [ord('q'), 27] # q Esc - Keys that closes showed windows
WINDOW_INFO_NAME = "Document Scanner: Esc/q -> exit | w -> save all components | e -> save final img | r -> save final img gray" # Window name that functions as info text
TEXT_COL = (255, 255, 255) # Color of text on info popup
POS_INFO = (45, 134, 45) # Color of positive info popup
NEG_INFO = (0, 51, 255) # Color of negative info popup
WARN_INFO = (0, 153, 255) # Color of warning info popup

# Component file name end strings that will be added to original image name when saved
SCALED_IMG_FILENAME = "1_scaled"
GRAY_IMG_FILENAME = "2_gray"
BLUR_IMG_FILENAME = "3_blurred"
EDGE_IMG_FILENAME = "4_edges"
CONS_IMG_FILENAME = "5_contours"
CON_IMG_FILENAME = "6_contour"
POINTS_IMG_FILENAME = "7_points"
TRANSFORMED_IMG_FILENAME = "8_transformed"
BALANCED_IMG_FILENAME = "9_balanced"
BALANCED_IMG_GRAY_FILENAME = "10_balanced_gray"
FINAL_IMG = "final"
FINAL_IMG_GR = "final_bw"

def check_arguments(img_path: str, save_folder_path: str):
    """Check whether given arguments are correct file and results folder paths.

    Args:
        img_path (str): Image file path and name.
        save_folder_path (str): Folder path where results will be saved to.
    """
    if not os.path.isfile(img_path):
        print(f"Error: The file '{img_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(save_folder_path):
        try:
            os.makedirs(save_folder_path)
        except Exception as e:
            print(f"Error: Could not create the folder '{save_folder_path}': {e}")
            sys.exit(1)

def resize_img(img: np.ndarray, max_len: int=MAX_LEN_IMG) -> np.ndarray:
    """Resize image shape proportional to max_len if width or height is larger.

    Args:
        img (np.ndarray): Image to be checked and resized if needed.
        max_len (int, optional): Maximum length than an image width or height can have. Defaults to MAX_LEN_IMG.

    Returns:
        np.ndarray: Image with width, height less or equal to max_len.
    """
    rows, cols, _ = img.shape
    longest = max(rows, cols)

    if longest > max_len:
        new_size = (int(cols / longest * max_len), int(rows / longest * max_len))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return img

def fill_images_with_black(images: dict, ref_img: np.ndarray) -> dict:
    """Fill all values of images dict with default black images that have shape of reference image.

    Args:
        images (dict): Dictionary that will contain np.ndarray images as values with its names as keys.
        ref_img (np.ndarray): Reference image.

    Returns:
        dict: Filled dictionary with default black images.
    """
    black_image = np.zeros_like(ref_img)
    filled_images = {name: black_image.copy() for name in images.keys()}

    return filled_images

def get_max_contour(img: np.ndarray, contours: np.ndarray) -> np.array:
    """Get the biggest contour that is rectangle-like and with size of at least 30% of image.

    Args:
        img (np.ndarray): Input image.
        contours (np.ndarray): The contours of the image.

    Returns:
        np.array: Biggest contour that meets requirements, returns None if no contour meets the specified requirements.
    """
    max_con, best_area = np.array([]), 0
    min_area = (img.shape[0] * img.shape[1]) * (MIN_CONTOUR_AREA_PERC / 100)

    for con in contours:
        area = cv2.contourArea(con)
        if area > min_area and area > best_area:
            approx = cv2.approxPolyDP(con, 0.02 * cv2.arcLength(con, True), True)
            if len(approx) == 4:
                max_con = approx
                best_area = area

    return max_con

def sort_points(points: np.ndarray) -> np.ndarray:
    """Sort the input points representing the corners in a specific order.

    Args:
        points (np.ndarray): Input array of points to be sorted.

    Returns:
        np.ndarray: Array with shape (4, 1, 2) containing the sorted points.
    """
    points = points.reshape((4, 2)).astype(np.float32)
    points_sorted = np.zeros((4, 1, 2), dtype=np.float32)

    points_diff = np.diff(points, axis=1)
    points_sum = points.sum(1)
    points_sorted[0] = points[np.argmin(points_sum)]
    points_sorted[1] = points[np.argmin(points_diff)]
    points_sorted[2] = points[np.argmax(points_diff)]
    points_sorted[3] = points[np.argmax(points_sum)]

    return points_sorted

def transform_perspective_img(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform the perspective of an input image based on the specified points.

    Args:
        img (np.ndarray): The input image as a np.ndarray.
        points (np.ndarray): The four corner points of the region to be transformed.

    Returns:
        np.ndarray: The transformed image.
    """
    height, width, _ = img.shape
    points = sort_points(points)
    points_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(points, points_2)
    img_transformed = cv2.warpPerspective(img, matrix, (width, height))

    return img_transformed

def crop_img(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop image by crop_size from each side and resize cropped image to original image shape.

    Args:
        img (np.ndarray): Image to be cropped.
        crop_size (int): Amount of pixels to be cropped from each side.

    Returns:
        np.ndarray: Cropped and resized image.
    """
    cropped_img = img[crop_size:-crop_size, crop_size:-crop_size]
    resized_cropped_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    return resized_cropped_img

def auto_balance_img_col(img: np.ndarray) -> np.ndarray:
    """Perform auto-balancing on the input image.

    Args:
        img (np.ndarray): Image to be auto-balanced.

    Returns:
        np.ndarray: The normalized image after auto-balancing.
    """
    normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return normalized_img

def auto_balance_img_white(img: np.ndarray) -> np.ndarray:
    """Perform auto-balancing on the input image to be black and white.

    Args:
        img (np.ndarray): Image to be auto-balanced.

    Returns:
        np.ndarray: Image that is black and white.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)
    balanced_img = cv2.medianBlur(adaptive_img, 3)

    return balanced_img

def show_image_on_postion(img: np.ndarray, win_name: str="Image", x: int=10, y: int=10, width: int=1400, height: int=800) -> str:
    """Show given image in a window specified at (x, y) position with shape (width, height)

    Args:
        img (np.ndarray): Image to be shown.
        win_name (str, optional): Name of the window. Defaults to "Image".
        x (int, optional): Horizontal position of the beggining of thr window. Defaults to 10.
        y (int, optional): Vertical position of the beggining of the window. Defaults to 10.
        width (int, optional): Width of the window. Defaults to 1400.
        height (int, optional): Height of the window. Defaults to 800.

    Returns:
        str: Name of the window for future reference.
    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, x, y)
    cv2.resizeWindow(win_name, width, height)
    cv2.imshow(win_name, img)

    return win_name

def save_img(img: np.ndarray, file_name: str, folder_path, base_img_name: str, file_format: str) -> bool:
    """Save the image to a file in the specified folder. Return boolean status of operation.

    Args:
        img (np.ndarray): Image to be saved.
        file_name (str): The name of the file to be saved.
        folder_path (os.path, optional): Folder where the file should be saved.
        base_img_name (str): Original image file name for image save file naming.
        file_format (str): Original image file format.

    Returns:
        bool: Boolean value that indicates status of operation. True is successfull, False otherwise if image is empty.
    """
    full_file_name = f"{base_img_name}_{file_name}{file_format}"
    file_path = os.path.join(folder_path, full_file_name)

    if np.all(img == 0):
        return False

    try:
        cv2.imwrite(file_path, img)
        return True
    except Exception as e:
        print(f"Error: saving image to file: {e}")
        sys.exit(1)

def save_images(images: dict, folder_path: os.path, base_img_name: str, file_format: str):
    """Save images given in dictionary.

    Args:
        images (dict): Dictionary of np.ndarray type images.
        folder_path (os.path, optional): Folder where the file should be saved.
        img_file_name (str): Original image file name for image save file naming.
        file_format (str): Original image file format.
    """
    return all(save_img(img, file_name, folder_path, base_img_name, file_format) for file_name, img in images.items())

def add_info_on_window(win_name: str, txt: str, img: np.ndarray, wait_time: int=1500, txt_color=TEXT_COL, bg_color = POS_INFO):
    """
    Add text to the middle of the specified window and sleep for certain amount of time, then show original image without text.

    Args:
        win_name (str): Name of the window.
        txt (str): Text to be added.
        img (np.ndarray): Image associated with the window.
        wait_time (int): Amount of time in miliseconds to wait before added text dissapears. Defaults 1500.
        txt_color : Color of the text.
        bg_color : Color of the text background.
    """
    img_copy = img.copy()

    font_thickness = 40
    font_scale = min(img_copy.shape[1] // (len(txt) * font_thickness), img_copy.shape[0] // 10)
    txt_size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    txt_position = ((img_copy.shape[1] - txt_size[0]) // 2, (img_copy.shape[0] + txt_size[1]) // 2)

    rectangle_size = (txt_size[0] + font_thickness*5, txt_size[1] + font_thickness*5)
    rectangle_position = ((img_copy.shape[1] - rectangle_size[0]) // 2, (img_copy.shape[0] + rectangle_size[1]) // 2)
    cv2.rectangle(img_copy, rectangle_position, (rectangle_position[0] + rectangle_size[0], rectangle_position[1] - rectangle_size[1]), bg_color, thickness=cv2.FILLED)

    cv2.putText(img_copy, txt, txt_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, font_thickness)
    cv2.imshow(win_name, img_copy)
    cv2.waitKey(wait_time)
    cv2.imshow(win_name, img)

def process_a4(img_path: str, folder_save: str, base_img_name: str, file_format: str):
    """Process input image and show component images. Run infinite loop that checks user input.

    Args:
        img_path (str): Path to original image file.
        folder_save (str): Folder location where results should be saved.
        base_img_name (str): Original image file name.
        file_format (str): Original image file extension.
    """
    img = cv2.imread(img_path)
    img_scaled = resize_img(img)

    images = {
        SCALED_IMG_FILENAME         : None,
        GRAY_IMG_FILENAME           : None,
        BLUR_IMG_FILENAME           : None,
        EDGE_IMG_FILENAME           : None,
        CONS_IMG_FILENAME           : None,
        CON_IMG_FILENAME            : None,
        POINTS_IMG_FILENAME         : None,
        TRANSFORMED_IMG_FILENAME    : None,
        BALANCED_IMG_FILENAME       : None,
        BALANCED_IMG_GRAY_FILENAME  : None,
    }
    images = fill_images_with_black(images, img_scaled)
    images[SCALED_IMG_FILENAME] = img_scaled

    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    images[GRAY_IMG_FILENAME] = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1)
    images[BLUR_IMG_FILENAME] = cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2BGR)
    img_edges = cv2.Canny(img_blurred, 30, 200) #TODO real time edge sliders with spacebar confirmation
    images[EDGE_IMG_FILENAME] = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cons = cv2.drawContours(img_scaled.copy(), contours, -1, (0, 255, 0), 10)
    images[CONS_IMG_FILENAME] = img_cons

    max_contour = get_max_contour(img_scaled, contours)
    if max_contour.size:
        img_con = cv2.drawContours(img_scaled.copy(), [max_contour], 0, (0, 0, 255), 10)
        images[CON_IMG_FILENAME] = img_con

        points = max_contour.reshape(-1, 2)
        img_points = img_con.copy()
        for point in points:
            cv2.circle(img_points, point, 15, (255, 0, 0), -1)
        images[POINTS_IMG_FILENAME] = img_points

        img_transformed = transform_perspective_img(img_scaled, points)
        images[TRANSFORMED_IMG_FILENAME] = img_transformed
        img_cropped = crop_img(img_transformed, CROP_SIZE)
        img_balanced = auto_balance_img_col(img_cropped)
        images[BALANCED_IMG_FILENAME] = img_balanced
        img_balanced_gray = auto_balance_img_white(img_cropped)
        images[BALANCED_IMG_GRAY_FILENAME] = cv2.cvtColor(img_balanced_gray, cv2.COLOR_GRAY2BGR)

    images_amount_half = len(images) // 2
    images_values = list(images.values())
    row_1 = np.hstack(images_values[:images_amount_half])
    row_2 = np.hstack(images_values[images_amount_half:])
    stacked = np.vstack((row_1, row_2))
    win_name = show_image_on_postion(stacked, WINDOW_INFO_NAME)

    while True:
        key = cv2.waitKey(1)
        if key in KEY_CLOSERS or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            break

        status, message, bg_color = None, None, None

        if key == ord('w'):
            components_save = os.path.join(folder_save, "saved_components")
            if not os.path.exists(components_save):
                try:
                    os.makedirs(components_save)
                except Exception as e:
                    print(f"Error: Could not create the '{components_save}' folder: {e}")
                    sys.exit(1)

            status = save_images(images, components_save, base_img_name, file_format)
            message = "SAVED SUCCESSFULLY" if status else "SAVED PARTIALLY"
        elif key == ord('e'):
            status = save_img(images[BALANCED_IMG_FILENAME], FINAL_IMG, folder_save, base_img_name, file_format)
            message = "SAVED SUCCESSFULLY" if status else "NO IMAGE TO SAVE"
        elif key == ord('r'):
            status = save_img(images[BALANCED_IMG_GRAY_FILENAME], FINAL_IMG_GR, folder_save, base_img_name, file_format)
            message = "SAVED SUCCESSFULLY" if status else "NO IMAGE TO SAVE"

        if status is not None:
            bg_color = POS_INFO if status else (NEG_INFO if "NO IMAGE" in message else WARN_INFO)
            add_info_on_window(win_name, message, stacked, bg_color=bg_color)

def main():
    if len(sys.argv) not in {2, 3}:
        print("Usage: python paper_scan.py <img_path> <save_folder_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    folder_save = sys.argv[2] if len(sys.argv) == 3 else FOLDER_SAVE

    check_arguments(img_path, folder_save)
    base_img_name, file_format = os.path.splitext(os.path.basename(img_path))
    process_a4(img_path, folder_save, base_img_name, file_format)

if __name__ == "__main__":
    main()