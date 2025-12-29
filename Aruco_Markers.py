import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import yaml

#create videocapture object
video = cv.VideoCapture(1)

#check if the video file was opened successfully
if not video.isOpened():
    print("Error: Could not open video")
    exit()

# Load calibration data with Python's yaml
with open(r"calibration.yaml") as f:
    calib = yaml.safe_load(f)
 
""" 
load your camera’s intrinsic matrix and distortion coefficients from a YAML file
- camera_matrix is the 3×3 intrinsic parameters matrix.
- dist_coeffs holds lens distortion parameters.
- marker_length is the side length of the ArUco marker in meters.
"""
camera_matrix = np.array(calib["camera_matrix"])
dist_coeffs = np.array(calib["dist_coeff"])
marker_length = 0.093 #m

#define corner obj pts
obj_points = np.array([
            [-marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2,  marker_length / 2, 0],
            [ marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float32)

#define dictionary
aruco_dict = aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
#define parameters
aruco_params = aruco.DetectorParameters()
#create detector object
aruco_detector = aruco.ArucoDetector(aruco_dict,aruco_params)

#display video
while True:
    ret,frame = video.read()

    if ret == False:
        print("error")
        break
    
    # Convert the frame to grayscale, which is more efficient for detection
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #detect marker
    corners,id,rejected = aruco_detector.detectMarkers(gray_frame)

    if id is not None:
        aruco.drawDetectedMarkers(frame,corners,id)

        for img_points in corners:
            retval,rvec,tvec = cv.solvePnP(obj_points,img_points,camera_matrix,dist_coeffs)

            if retval:
                # Draw the axis on the frame
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                 
                # Extract the translation vector and calculate the distance
                x, y, z = tvec.flatten()
                distance = np.linalg.norm(tvec)

                # Print to terminal
                print(f"X={x:.3f} m, Y={y:.3f} m, Z(depth)={z:.3f} m, Distance={distance:.3f} m")

                # Display on frame with different colors
                org = (20, 40)  
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                cv.putText(frame, f"X={x:.3f} m", (org[0], org[1]), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)
                cv.putText(frame, f"Y={y:.3f} m", (org[0], org[1]+30), font, font_scale, (255, 0, 0), thickness, cv.LINE_AA)
                cv.putText(frame, f"Depth={z:.3f} m", (org[0], org[1]+60), font, font_scale, (0, 0, 255), thickness, cv.LINE_AA)
                cv.putText(frame, f"Dist={distance:.3f} m", (org[0], org[1]+90), font, font_scale, (0, 255, 255), thickness, cv.LINE_AA)

    cv.imshow("Video",frame)

    # Press 'q' to exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
video.release()
cv.destroyAllWindows()
