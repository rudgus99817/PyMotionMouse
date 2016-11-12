from src import Mouse_Control
import cv2
from Study import edge
import numpy as np

cv2.namedWindow("Window")
cv2.namedWindow("Histogram")

process = edge.ImageProcess()
mouse = Mouse_Control.Control()

prior = (350, 187)
current = (350, 187)
p_finger = 5

# Capture Object Create
CameraCapture = cv2.VideoCapture(0)
success = True

# Main logic start
run = 0
track_Window = (100, 60, 200, 230)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while success:
    prior = current
    success, frame = CameraCapture.read(0)
    frame = process.resize(frame)
    if run < 100:
        run += 1
        process.draw_histo_rect(frame)
        cv2.putText(frame, 'Fingers', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_4)
        cv2.imshow("Histogram", frame)
        if cv2.waitKey(60) >= 27:
            break

        if run == 100:
            process.build_histogram(frame)
        continue

    elif run < 200:
        run += 1
        process.draw_histo_rect(frame)
        cv2.putText(frame, 'Palm', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_4)
        cv2.imshow("Histogram", frame)
        if cv2.waitKey(60) >= 27:
            break

        if run == 200:
            process.build_histogram(frame)
            dst = process.getBackprojection(frame)
            ret, track_Window = cv2.meanShift(dst, track_Window, term_crit)
            x, y, w, h = track_Window
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            current = prior = (x, y)
            mouse.move(prior, current)
        continue
    dst = process.getBackprojection(frame)

    # Setup the termination criteria, either 10 iteration or e by at least 1 pt
    ret, track_Window = cv2.meanShift(dst, track_Window, term_crit)
    x, y, w, h = track_Window
    current = (x, y)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    part = frame[y:y+h, x:x+w]
    part = process.processing(part)
    high, low = process.setrange()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))


    color = cv2.inRange(part, low, high)
    color = cv2.morphologyEx(color, cv2.MORPH_OPEN, kernel, iterations=2)
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel, iterations=1)
    part = cv2.bitwise_and(part, part, mask=color)

    centroid = None
    centroid, color = process.label(color)
    if centroid is not None:
        centroid = (int(centroid[0]), int(centroid[1]))
        cv2.circle(part, centroid, 3, [255, 0, 0])
        # current = centroid
        cv2.rectangle(color, (centroid[0]-65, centroid[1]-90), (centroid[0]+75, centroid[1]+120), [255, 255, 255], -1)
        # cv2.circle(color, (centroid[0], centroid[1]+10), 90, [255, 255, 255], -1)

    color, contours, ret = cv2.findContours(color, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    finger = len(contours) - 2
    ci = 0
    area = 0
    found = False
    cv2.drawContours(frame, contours, -1, [255, 0, 0], 3)
    if(finger == 1):
        mouse.move(prior, current)
    if(p_finger == 3 and finger < 3):
        mouse.click()
    if(p_finger == 4 and finger < 3):
        mouse.double_click()
    p_finger = finger

    cv2.putText(frame, str(finger), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_4)

    part = cv2.cvtColor(part, cv2.COLOR_HSV2BGR)
    cv2.imshow("Window", frame)
    cv2.imshow("Histogram", part)

    if cv2.waitKey(60) >= 27:
        break

CameraCapture.release()
cv2.destroyAllWindows()