import cv2
import Image_process
import Mouse_Control

cv2.namedWindow("Window")
cv2.namedWindow("Histogram")

process = Image_process.ImageProcess()
mouse = Mouse_Control.Control()
clicked = False

prior = (350, 187)
current = (350, 187)

#Capture Object Create
CameraCapture = cv2.VideoCapture(0)
success = True

#Main logic start
run = 0
track_Window = (100, 60, 200, 200)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(success):
    prior = current
    success, frame = CameraCapture.read(0)
    frame = process.resize(frame)
    cv2.GaussianBlur(frame, (3, 3), 0, frame)

    if(run < 100):
        run += 1
        process.Draw_histo_rect(frame)
        cv2.imshow("Histogram", frame)
        if(cv2.waitKey(60) >= 27):
            break

        if(run == 100):
            process.Build_histogram(frame)
            dst, mask = process.Apply_histo_mask(frame)
            ret, track_Window = cv2.meanShift(dst, track_Window, term_crit)
            x, y, w, h = track_Window
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            current = prior = (x, y)
            mouse.move(prior, current)
        continue

    dst, mask = process.Apply_histo_mask(frame)

    # Setup the termination criteria, either 10 iteration or e by at least 1 pt
    ret, track_Window = cv2.meanShift(dst, track_Window, term_crit)
    x, y, w, h = track_Window
    current = (x, y)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    mouse.move(prior, current)
    frame = cv2.bitwise_and(frame, mask)

    part = frame[y:y+3, x:x+w]
    mask = mask[y:y+3, x:x+w]
    part = cv2.bitwise_and(part, mask)

    if(part.max() == 0):
        clicked = True
    else:
        clicked = False


    if(mouse.clickstat(clicked)):
        #num = process.Labeling(part)
        #print(num)
        mouse.click()

    cv2.imshow("Window", frame)
    cv2.imshow("Histogram", part)

    if(cv2.waitKey(60) >= 27):
        break

CameraCapture.release()
cv2.destroyAllWindows()