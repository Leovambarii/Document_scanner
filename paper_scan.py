import cv2
import sys
import os
import numpy as np

#TODO add sliders
#TODO add save options hotkeys
#TODO add edition loop
#TODO clean code into functions
#TODO add error files checks + add dedicated folder for images
FOLDER_SAVE = "results"
MAX_LEN_IMG = 2048 # Maximum width or height that orginal image will be proportionally resized
CROP_SIZE = 25 # Amount of pixels that will be cropped from each side of image for final image
KEY_CLOSERS = [ord('q'), 27, 32] # q Esc Space - Keys that closes shown windows
WINDOW_INFO_NAME = "Document Scanner: Esc/q -> exit | s -> save all components | Spacebar -> save final img"

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

def get_contour_edge_points(contour: np.ndarray) -> np.ndarray:
    """Get the edge points of a contour by approximating its shape.

    Args:
        contour (np.ndarray): The input contour.

    Returns:
        np.ndarray: An array containing the edge points of the approximated contour.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx.reshape(-1, 2)

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

def auto_balance_img(image: np.ndarray) -> np.ndarray: #TODO edit for better results
    """Perform auto-balancing on the input image.

    Args:
        image (np.ndarray): Image to be auto-balanced.

    Returns:
        np.ndarray: The normalized image after auto-balancing.
    """
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return normalized_image

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

def save_img(img: np.ndarray, file_name: str, folder_path: os.path=FOLDER_SAVE):
    """Save the image to a file in the specified folder.

    Args:
        img (np.ndarray): Image to be saved.
        file_name (str): The name of the file to be saved.
        folder_path (os.path, optional): Folder where the file should be saved. Defaults to FOLDER_SAVE.
    """
    cv2.imwrite(os.path.join(folder_path, file_name), img)
    try:
        cv2.imwrite(os.path.join(folder_path, file_name), img)
    except Exception as e:
        print(f"Error: saving image to file: {e}")
        sys.exit(1)

def save_images(images: dict, folder_path: os.path=FOLDER_SAVE):
    """Save images given in dictionary.

    Args:
        images (dict): Dictionary of np.ndarray type images.
        folder_path (os.path, optional): Folder where the file should be saved. Defaults to FOLDER_SAVE.
    """
    for file_name, img in images.items():
        save_img(img, file_name, folder_path)

def process_a4(img_path):
    img = cv2.imread(img_path)
    img_scaled = resize_img(img)
    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_edges = cv2.Canny(img_blurred, 30, 200) #TODO real time edge sliders with spacebar confirmation

    contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cons = cv2.drawContours(img_scaled.copy(), contours, -1, (0, 255, 0), 10)
    max_contour = max(contours, key=cv2.contourArea) # TODO add case when there is missing contour
    img_con = cv2.drawContours(img_scaled.copy(), [max_contour], 0, (0, 0, 255), 10)

    points = get_contour_edge_points(max_contour)
    img_points = img_con.copy()
    for point in points:
        cv2.circle(img=img_points, center=point, radius=15, color=(255, 0, 0), thickness=-1)

    img_transformed = transform_perspective_img(img=img_scaled, points=points)
    img_cropped = crop_img(img_transformed, CROP_SIZE)
    img_balanced_gray = auto_balance_img(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY))

    images = {  "img_1_scaled.jpg"      :   img_scaled,
                "img_2_gray.jpg"        :   cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
                "img_3_blurred.jpg"     :   cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2BGR),
                "img_4_edges.jpg"       :   cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR),
                "img_5_contours.jpg"    :   img_cons,
                "img_6_contour.jpg"     :   img_con,
                "img_7_points.jpg"      :   img_points,
                "img_8_transformed.jpg" :   img_transformed,
                "img_9_cropped.jpg"     :   img_cropped,
                "img_10_balanced.jpg"   :   cv2.cvtColor(img_balanced_gray, cv2.COLOR_GRAY2BGR)
    }
    images_amount_half = len(images) // 2
    images_values = list(images.values())
    row_1 = np.hstack(images_values[:images_amount_half])
    row_2 = np.hstack(images_values[images_amount_half:])
    stacked = np.vstack((row_1, row_2))
    win_name = show_image_on_postion(img=stacked, win_name=WINDOW_INFO_NAME)

    while True:
        key = cv2.waitKey(1)
        if key in KEY_CLOSERS or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            break
        if key == ord('s'):
            save_images(images=images)

def main():
    if len(sys.argv) not in {2, 3}:
        print("Usage: python paper_scan.py <img_path> <save_folder_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if len(sys.argv) == 3:
        FOLDER_SAVE = sys.argv[2]

    check_arguments(img_path, FOLDER_SAVE)
    process_a4(img_path)

if __name__ == "__main__":
    main()