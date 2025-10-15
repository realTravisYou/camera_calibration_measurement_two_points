import cv2
import numpy as np
import pickle

# ====== 1. 加载相机标定参数 ======
with open("camera_cali_params_0916.pkl", "rb") as f:
    calib_data = pickle.load(f)

cameraMatrix = calib_data["cameraMatrix"]
distCoeffs = calib_data["dist"]
H = calib_data["H"]


print("已加载单应性矩阵:\n", H)

# ====== 3. 读取测量图像 ======
measure_img = cv2.imread("pen.bmp")
img_display = measure_img.copy()

# ====== 4. 像素坐标映射到世界坐标 ======
def pixel_to_world(u, v, H):
    pixel = np.array([u, v, 1]).reshape(3, 1)
    world = H @ pixel
    world /= world[2]
    return world[0][0], world[1][0]

# ====== 5. 计算物理距离 ======
def compute_distance(p1, p2):
    pts = np.array([p1, p2], dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, cameraMatrix, distCoeffs, P=cameraMatrix)
    u1, v1 = undistorted[0][0]
    u2, v2 = undistorted[1][0]

    X1, Y1 = pixel_to_world(u1, v1, H)
    X2, Y2 = pixel_to_world(u2, v2, H)

    distance = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
    print(f"物理距离: {distance:.3f} mm")

    cv2.line(img_display, p1, p2, (255, 0, 0), 2)
    cv2.putText(img_display, f"{distance:.1f} mm",
                ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("image", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ====== 6. 手动定义两点像素坐标 ======
p1 = (573, 250)   # ← 第一个点
p2 = (573, 846)   # ← 第二个点

# ====== 7. 执行计算 ======
compute_distance(p1, p2)
