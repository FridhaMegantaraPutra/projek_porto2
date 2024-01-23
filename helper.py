from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube

import settings
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import app


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model
