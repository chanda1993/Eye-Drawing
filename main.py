import cv2
import Detector
import GUI
import Homography
import numpy as np

# Threshold value for eyes pupil >50 fading <50 much clear depending on lighting
PUPIL_THRESH = 50
# PHASE 0: Pupils configuration # PHASE 1: Eyes Calibration # PHASE 2: Paint Mode
PHASE = 0
# Initial positioning of a mouse cursor before drawing
cursor_pos = [-1, -1]
green_img = np.tile(np.uint8([0, 255, 0]), (10, 10, 1))

if __name__ == '__main__':
    detector = Detector.CascadeDetector()
    gui = GUI.GUI()
    homo = Homography.Homography()

    # Access the webcam to capture the images
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow('Eyes Drawing Tool', cv2.WINDOW_AUTOSIZE)
    # cv2.createTrackbar('Threshold', 'Eyes Drawing Tool', 0, 255, gui.on_trackbar)
    # cv2.setTrackbarPos('Threshold', 'Eyes Drawing Tool', PUPIL_THRESH)

    # Keep the webcam running until eyes are detected
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        detector.find_eyes(frame)

        # Interchange the configurations UIs
        if PHASE == 1:
            if homo.homography is not None:
                PHASE = 2
                detector.start_phase(2)
                gui.end_calibration()
            else:
                if gui.calib_step(detector.left_is_visible, detector.right_is_visible):
                    homo.save_calibration_position([detector.left_pupil, detector.right_pupil],
                                                   gui.calibration_cursor_pos,
                                                   gui.calibration_counter)
                if gui.phase == 2:
                    PHASE = 2
                    homo.calculate_homography()
                    detector.start_phase(2)
        elif PHASE == 2:
            cursor_pos = homo.get_cursor_pos([detector.left_pupil, detector.right_pupil])

        gui.make_window(frame, detector.get_images(), cursor_pos, detector.overlap_threshold)

        # TODO draw eyes in the face_frame
        k = cv2.waitKey(33)
        if k == 27 & 0xFF == ord('q'):
            break
        elif k == 32:
            if PHASE < 2:
                if PHASE == 0:  # 0
                    if not detector.left_is_visible or not detector.right_is_visible:
                        gui.alert_box("Error", "Show both your eyes to the camera.")
                        detector.phase -= 1
                        gui.phase -= 1
                        PHASE -= 1
                    else:
                        # detector.phase += 1
                        # gui.phase += 1
                        # PHASE+=1
                        gui.alert_box("Calibration Phase", "Keep still your shoulders and follow the circle with "
                                                           "the eyes, moving with your head as more as possible.")
                        cv2.destroyWindow("Eyes Drawing Tool")
                        cv2.namedWindow('Eyes Drawing Tool', cv2.WINDOW_GUI_EXPANDED)
                        gui.run_calibration()
                else:
                    gui.alert_box("Paint Phase", "Keep still your shoulders and move the cursor with your eyes, "
                                                 "changing between drawing/pointing mode with space key. "
                                                 "Personalize the cursor and change the color by pressing the "
                                                 "relative key on the lateral bar.")
                detector.phase += 1
                gui.phase += 1
                PHASE += 1
            else:
                gui.toggle_drawing_mode()
        elif k == 60:  # < => decrease sensibility
            detector.overlap_threshold -= 0.01
        elif k == 62:  # > => increase sensibility
            detector.overlap_threshold += 0.01
        elif k == 105:  # i => Info on sensibility
            gui.alert_box("Info - Sensibility",
                          "Set the eyes detector sensibility: stop when the purple squares around the eyes are "
                          "stable but also they keep following the eyes smoothly.")

        if PHASE == 1:
            if k == 116:  # t => Info on threshold
                gui.alert_box("Info - Threshold",
                              "Set the pupils detector sensibility: stop when eyes and pupils are stably seen and "
                              "drawn.")

        if PHASE == 2:
            gui.check_key(k)

    cap.release()
    cv2.destroyAllWindows()
