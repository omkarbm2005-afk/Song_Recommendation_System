import cv2
import pygame
import numpy as np
import webbrowser
import time
from fer import FER

# initialize webcam
cap = cv2.VideoCapture(0)

# emotion detector
detector = FER()

# initialize pygame mixer
pygame.mixer.init()

emotion_detected = False

while True:

    ret, frame = cap.read()

    if not emotion_detected:

        result = detector.detect_emotions(frame)

        if len(result) > 0:

            face = result[0]

            (x, y, w, h) = face["box"]
            emotions = face["emotions"]

            # detect dominant emotion
            emotion = max(emotions, key=emotions.get)
            emotion = emotion.lower().strip()

            print("Detected Emotion:", emotion)

            # draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # display emotion text
            cv2.putText(frame, f"Emotion: {emotion}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

            cv2.imshow("Emotion Detection", frame)

            # show emotion for 2 seconds
            cv2.waitKey(2000)

            # open Spotify playlists based on emotion

            if emotion == "happy":
                webbrowser.open("https://open.spotify.com/playlist/7aizk517RO4gEtSfSUR2Qu?si=AFIivSeHSd6Rdq_jngZS-w")

            elif emotion == "sad":
                webbrowser.open("https://open.spotify.com/playlist/2UAc47TBlq1NrvWf3gHrsc?si=4717cba0d5d54d93")

            elif emotion == "neutral":
                webbrowser.open("https://open.spotify.com/playlist/13UFqyClP15LT0yCyEAX0E?si=t86iwVuLTHOR7z2ALW5NgA")

            elif emotion == "surprise":
                webbrowser.open("https://open.spotify.com/playlist/4OnZWEfqKMlHWXgA7JTAkr?si=24665ac170a548b3")

            else:
                webbrowser.open("https://open.spotify.com")

            # close webcam window
            cv2.destroyAllWindows()

            time.sleep(1)

            emotion_detected = True
            break

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()