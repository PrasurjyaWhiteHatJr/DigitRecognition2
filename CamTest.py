import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os, ssl, time
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

#Starting The Camera

cap = cv2.VideoCapture(1)

while(True):
    try:
        ret, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()