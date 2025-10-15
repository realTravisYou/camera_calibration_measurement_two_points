import numpy as np
import cv2 as cv
import glob
import pickle



chessboardSize = (11, 8)  # 表示棋盘格内部角点的数量（不是方格数量）
frameSize = (1624, 1240)  # 输入图像的分辨率

# termination criteria，终止条件，用于亚像素级角点优化（cv.cornerSubPix()）的停止标准。迭代最多 30 次或者角点位置变化小于 0.001 时停止。
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points，准备真实世界坐标
# 生成棋盘格角点在真实世界坐标系中的位置（Z=0 平面上），最终得到一个形状为 (88, 3) 的坐标矩阵。
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 10  # 每个格子大小为 10 mm（这个是棋盘格小格子的边长）
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.准备列表来存储所有图像的真实世界坐标和像素坐标。
objpoints = []  # 3d point in real world space在真实世界中的3D点坐标
imgpoints = []  # 2d points in image plane.在像素平面的2D点坐标

images = glob.glob('chessboardImages0916/*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners找到棋盘格的角点
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners显示找到的棋盘格角点
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(200)

cv.destroyAllWindows()

############## CALIBRATION 标定#######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print(f"ret = \n", ret)
print(f"camera matrix = \n", cameraMatrix)
print(f"dist = \n", dist)

############## COMPUTE HOMOGRAPHY (using one calibration image) 使用一张棋盘格平放在测量平面的图像来计算单应性矩阵###################

# 选择一张用于计算单应性的标定图
calib_img_path = r"chessboardImages0916/Image_20250916092808212.jpg"
img = cv.imread(calib_img_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
if not ret:
    raise RuntimeError(f"在 {calib_img_path} 中未检测到棋盘格角点")

corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# 世界坐标 (Z=0)
objp_h = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp_h[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp_h = objp_h * size_of_chessboard_squares_mm
objp_h_2d = objp_h[:, :2]

# 计算单应性矩阵
H, _ = cv.findHomography(corners2.reshape(-1, 2), objp_h_2d)
print("单应性矩阵 H = \n", H)

############## SAVE CALIBRATION DATA (包括 H) 保存标定结果（包括 H 单应性矩阵）#####################################

with open("camera_cali_params_0916.pkl", "wb") as f:
    pickle.dump({
        "cameraMatrix": cameraMatrix,
        "dist": dist,
        "H": H
    }, f)

print("标定参数（含单应性矩阵 H）已保存到 camera_cali_params_0916.pkl")

############## REPROJECTION ERROR 重投影误差 #################################################
# 计算“实际检测到的角点位置”和“根据标定参数投影回去的理论角点”之间的平均误差。这项指标可以判断标定精度。
# 通常 误差 < 0.3 像素 说明标定效果很好。
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total reprojection error: {}".format(mean_error / len(objpoints)))
