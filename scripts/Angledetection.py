import numpy as np
import cv2
import cv2.aruco as aruco


cap = cv2.VideoCapture(0)  # Get the camera source

# FILE_STORAGE_READ


cv_file = cv2.FileStorage("calib_images/calibrationCoefficients.yaml", cv2.FILE_STORAGE_READ)




    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()

    # Debug: print the values
    # print("camera_matrix : ", camera_matrix.tolist())
    # print("dist_matrix : ", dist_matrix.tolist())

cv_file.release()

    # Debug: print the values
    # print("camera_matrix : ", camera_matrix.tolist())
    # print("dist_matrix : ", dist_matrix.tolist())





def track(matrix_coefficients, distortion_coefficients):
    while True:

        ret,frame = cap.read()  # Get the frame
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  

        parameters = aruco.DetectorParameters_create()  # Marker detection parameters
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)                                

    # store rz1 and rz2
        R_list=[]

        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers

        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                   distortion_coefficients)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error



                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis


                R, _ = cv2.Rodrigues(rvec)
            # convert (np.matrix(R).T) matrix to array using np.squeeze(np.asarray()) to get rid off the ValueError: shapes (1,3) and (1,3) not aligned
                R = np.squeeze(np.asarray(np.matrix(R).T))
                R_list.append(R[2])


        # Display the resulting frame


        if len(R_list) == 2:


            print('##############')
            angle_radians = np.arccos(np.dot(R_list[0], R_list[1]))
            angle_degrees=angle_radians*180/np.pi
            print(angle_degrees)


        cv2.imshow('frame', frame)
#   Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3000) & 0xFF
        if key == ord('q'): break



track(camera_matrix, dist_matrix )