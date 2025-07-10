import numpy as np
import os
from PIL import Image
from sklearn.svm import SVC

def rgb_to_gray(image):
    return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

def resize(image, new_h, new_w):
    h, w = image.shape
    result = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            x = int(i * h / new_h)
            y = int(j * w / new_w)
            result[i, j] = image[x, y]
    return result

def compute_gradients(image):
    padded = np.pad(image, 1, mode='edge')
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gx[i, j] = padded[i + 1, j + 2] - padded[i + 1, j]
            gy[i, j] = padded[i + 2, j + 1] - padded[i, j + 1]
    magnitude = np.sqrt(gx2 + gy2)
    orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    return magnitude, orientation

def compute_histograms(mag, ori, cell_size=8, bins=9):
    h, w = mag.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    hist = np.zeros((cells_y, cells_x, bins))
    bin_width = 180 / bins

    for i in range(cells_y):
        for j in range(cells_x):
            for u in range(cell_size):
                for v in range(cell_size):
                    y = i * cell_size + u
                    x = j * cell_size + v
                    angle = ori[y, x]
                    magnitude = mag[y, x]
                    bin_idx = angle / bin_width
                    lower_bin = int(np.floor(bin_idx)) % bins
                    upper_bin = (lower_bin + 1) % bins
                    ratio = bin_idx - lower_bin
                    hist[i, j, lower_bin] += magnitude * (1 - ratio)
                    hist[i, j, upper_bin] += magnitude * ratio
    return hist

def normalize_blocks(hist, block_size=2):
    h, w, b = hist.shape
    blocks_y = h - block_size + 1
    blocks_x = w - block_size + 1
    hog = []
    for i in range(blocks_y):
        for j in range(blocks_x):
            block = hist[i:i+block_size, j:j+block_size, :].ravel()
            norm = np.sqrt(np.sum(block**2) + 1e-5)
            block = block / norm
            hog.extend(block)
    return np.array(hog)

def compute_hog(image_rgb):
    gray = rgb_to_gray(image_rgb)
    resized = resize(gray, 128, 64)
    mag, ori = compute_gradients(resized)
    hist = compute_histograms(mag, ori)
    hog_vector = normalize_blocks(hist)
    return hog_vector

def load_dataset(folder_path):
    features = []
    labels = []
    for label in ['cats', 'dogs', 'human']:
        class_path = os.path.join(folder_path, label)
        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
            hog_vector = compute_hog(img_np)
            features.append(hog_vector)
            labels.append(label)
    return features, labels

X_train, y_train = load_dataset("dataset/train")
clf = SVC(kernel='linear', max_iter=10000)
clf.fit(X_train, y_train)
X_test, y_test = load_dataset("dataset/test")

np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

with open("hog_output2.txt", "w") as hog_file:
    for i in range(len(X_test)):
        hog_file.write(f"True Label     : {y_test[i]}\n")
        hog_file.write(f"Predicted Label: {clf.predict([X_test[i]])[0]}\n")
        hog_file.write(f"HOG Vector Shape: {X_test[i].shape}\n")
        hog_file.write("HOG Vector:\n")
        hog_file.write(np.array2string(X_test[i], separator=', '))
        hog_file.write("\n*****\n")