import cv2
import sys
import numpy as np

#TODO add sliders
#TODO add save options hotkeys
CROP_SIZE = 25
MAX_LEN_IMG = 1000

def resize_img(input_img, max_len=MAX_LEN_IMG):
    rows, cols, _ = input_img.shape
    longest = max(rows, cols)

    if longest > max_len:
        new_size = (int(cols / longest * max_len), int(rows / longest * max_len))
        input_img = cv2.resize(input_img, new_size, interpolation=cv2.INTER_LINEAR)

    return input_img

def get_contour_edge_points(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx.reshape(-1, 2)

def sort_points(points):
    points = points.reshape((4, 2)).astype(np.float32)
    points_sorted = np.zeros((4, 1, 2), dtype=np.float32)

    points_diff = np.diff(points, axis=1)
    points_sum = points.sum(1)
    points_sorted[0] = points[np.argmin(points_sum)]
    points_sorted[1] = points[np.argmin(points_diff)]
    points_sorted[2] = points[np.argmax(points_diff)]
    points_sorted[3] = points[np.argmax(points_sum)]

    return points_sorted

def crop_img(img, crop_size):
    cropped_img = img[crop_size:-crop_size, crop_size:-crop_size] # Crop all sides of image by crop_size
    resized_cropped_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) # Resize image to previous shape

    return resized_cropped_img

def auto_balance_img(image):
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # Normalize the image using the mean and standard deviation

    return normalized_image

def show_image_on_postion(img, win_name='Image', x=100, y=100, width=1500, height=650):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, x, y)
    cv2.resizeWindow(win_name, width, height)
    cv2.imshow(win_name, img)

def process_a4(img_path):
    img = cv2.imread(img_path)
    cv2.imwrite("img_0_org.jpg", img)

    img_scaled = resize_img(img)
    cv2.imwrite("img_1_scaled.jpg", img_scaled)

    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("img_2_gray.jpg", img_gray)

    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1)
    cv2.imwrite("img_3_blurred.jpg", img_blurred)

    img_edges = cv2.Canny(img_blurred, 30, 200) #TODO real time edge sliders with spacebar confirmation
    cv2.imwrite("img_4_edges.jpg", img_edges)

    contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cons = img_scaled.copy()
    cv2.drawContours(img_cons, contours, -1, (0, 255, 0), 3)
    cv2.imwrite("img_5_contours.jpg", img_cons)

    max_contour = max(contours, key=cv2.contourArea)
    img_con = cv2.drawContours(img_scaled.copy(), [max_contour], 0, (0, 0, 255), 3)
    cv2.imwrite("img_6_contour.jpg", img_con)

    points = get_contour_edge_points(max_contour)
    point_color = (255, 0, 0)
    img_points = img_con.copy()
    for point in points:
        cv2.circle(img_points, point, 8, point_color, -1)

    cv2.imwrite("img_7_points.jpg", img_points)

    height, width = img_gray.shape
    points = sort_points(points)
    points_2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(points, points_2)
    img_transformed = cv2.warpPerspective(img_scaled, matrix, (width, height))
    cv2.imwrite("img_8_transformed.jpg", img_transformed)

    img_cropped = crop_img(img_transformed, CROP_SIZE)
    cv2.imwrite("img_9_cropped.jpg", img_cropped)

    img_balanced_gray = auto_balance_img(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY))
    cv2.imwrite("img_10_balanced.jpg", img_balanced_gray)

    row_1 = np.hstack([img_scaled, cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2BGR),  cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR), img_cons])
    row_2 = np.hstack([img_con, img_points, img_transformed, img_cropped,  cv2.cvtColor(img_balanced_gray, cv2.COLOR_GRAY2BGR)])
    stacked = np.vstack((row_1, row_2))
    show_image_on_postion(stacked, 'Images')

    key_closers = [ord('q'), 27, 32] # q Esc Space
    while True:
        k = cv2.waitKey(0)
        print(k)
        if k in key_closers:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python paper_scan.py <img_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    process_a4(img_path)
