from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import settings
import app
from app import penerima


def send_email(penerima):
    sender_email = "apialrm727@gmail.com"
    sender_password = "nvlm ynph suqb xfwl"
    receiver_email = penerima

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "API DETECTED"

    body = "SEARCH LOCATION"
    message.attach(MIMEText(body, "plain"))

    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()

    server.login(sender_email, sender_password)

    server.sendmail(
        sender_email, receiver_email, message.as_string())

    server.quit()
    print("berhasil helper")
