import concurrent
import concurrent.futures
import multiprocessing
import os.path
import shutil
import time

import cv2
from PIL import Image
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.face import FaceClient
from moviepy.editor import *
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import matplotlib.pyplot as plt
import numpy as np

slices = 0
KEY = "c1edebb90dee461b85fa73c96a7f1021"
ENDPOINT = "https://proyecto-so.cognitiveservices.azure.com/"

files_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "files")
images_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "img")
videos_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vid")
text_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "text")
cpu_quantity = multiprocessing.cpu_count()

emotions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

forbidden_content = [0, 0, 0]


def get_video_duration(video_clip_path):
    return VideoFileClip(video_clip_path).duration


def split_video(video_clip_path, init_time, end_time, portion):
    result = VideoFileClip(video_clip_path).subclip(init_time, end_time)
    result.write_videofile(f"spliced_videos/{portion}.mp4", threads=multiprocessing.cpu_count(), audio=False)
    result.close()


def split_video_multiprocessing(video_clip_path):
    slices = multiprocessing.cpu_count()
    slides_portion = get_video_duration(video_clip_path) / slices
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for x in range(0, slices):
            print(f"Init time: {x * slides_portion} End time: {(x + 1) * slides_portion} Name: {x}")
            result = executor.submit(split_video, video_clip_path, x * slides_portion, (x + 1) * slides_portion, x)


def split_video_threads(video_clip_path):
    slices = multiprocessing.cpu_count()
    slides_portion = get_video_duration(video_clip_path) / slices
    for x in range(0, slices):
        split_video(video_clip_path, x * slides_portion, (x + 1) * slides_portion, x)


def get_frame(sec, video_capture, thread, number):
    video_capture.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    has_frame, image = video_capture.read()
    if has_frame:
        cv2.imwrite(f"img/{thread}{str(number)}.jpg", image)
    return has_frame


def video_to_images(video_clip_path, thread):
    video_capture = cv2.VideoCapture(video_clip_path)
    sec = 0
    frame_rate = 5
    count = 0
    success = get_frame(sec, video_capture, thread, count)
    while success:
        count += 1
        sec = sec + frame_rate
        sec = round(sec, 2)
        success = get_frame(sec, video_capture, thread, count)
    return success


def video_to_images_threads():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for filename in os.listdir("spliced_videos"):
            executor.submit(video_to_images, f"spliced_videos/{filename}", f"THREAD{filename}-")


def process_videos():
    for video_path in os.listdir("vid"):
        print(os.path.join(video_path, video_path))
        split_video_multiprocessing(os.path.join(videos_folder, video_path))
    video_to_images_threads()


def clean(dir_):
    for f in os.listdir(dir_):
        os.remove(os.path.join(dir_, f))


def get_extension(file_name):
    return file_name.split(".")[len(file_name.split(".")) - 1]


def sort_files(file_name):
    if get_extension(file_name) == "jpg" or get_extension(file_name) == "png":

        shutil.move(os.path.join(files_folder, file_name), images_folder)
    elif get_extension(file_name) == "mp4":
        shutil.move(os.path.join(files_folder, file_name), videos_folder)
    elif get_extension(file_name) == "txt":
        shutil.move(os.path.join(files_folder, file_name), text_folder)
    else:
        print(f"File type not supported: {get_extension(file_name)}")


def get_packages(dir_name):
    portions = round(len(os.listdir(dir_name)) / cpu_quantity) + 1
    if len(os.listdir(dir_name)) <= portions:
        return [os.listdir(dir_name)]
    return [os.listdir(dir_name)[i:i + portions] for i in range(0, len(os.listdir(dir_name)), portions)]


def get_files_on_dir(file_names_list):
    for filename in file_names_list:
        sort_files(filename)


def sort_files_threads():
    # Se determina si la cantidad de archivos es menor que la cantidad de nucleos disponibles
    # En caso de que sea menor es mejor ejecutarlo de forma secuencial
    if len(os.listdir("files")) <= cpu_quantity * 2:
        get_files_on_dir(os.listdir("files"))
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for sl in get_packages("files"):
                executor.submit(get_files_on_dir, sl)


def images_detection_processing(img_list):
    for image_url in img_list:
        print(f"Processing {image_url}")
        image_object = open(os.path.join(images_folder, image_url), "rb")

        single_image_name = os.path.basename(image_url)
        face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

        detected_faces = face_client.face.detect_with_stream(image=image_object,
                                                             return_face_attributes=[
                                                                 'age',
                                                                 'gender',
                                                                 'headPose',
                                                                 'smile',
                                                                 'facialHair',
                                                                 'glasses',
                                                                 'emotion',
                                                                 'hair',
                                                                 'makeup',
                                                                 'occlusion',
                                                                 'accessories',
                                                                 'blur',
                                                                 'exposure',
                                                                 'noise'
                                                             ])
        if not detected_faces:
            raise Exception('No face detected from image {}'.format(single_image_name))

        print("---------------------------------------------")
        print(f"|For image name: {image_url}|")
        print("---------------------------------------------")

        for face in detected_faces:
            print(f"->     Azure id: {face.face_id}")
            print(f"->     Detected age: {face.face_attributes.age}")
            print(f"->     Detected gender: {face.face_attributes.gender}")
            print(f"->     Detected emotion: {face.face_attributes.emotion}")
            print(f"->     Anger: {face.face_attributes.emotion.anger}")
            emotions[0] = emotions[0] + face.face_attributes.emotion.anger
            emotions[1] = emotions[1] + face.face_attributes.emotion.contempt
            emotions[2] = emotions[2] + face.face_attributes.emotion.disgust
            emotions[3] = emotions[3] + face.face_attributes.emotion.fear
            emotions[4] = emotions[4] + face.face_attributes.emotion.happiness
            emotions[5] = emotions[5] + face.face_attributes.emotion.neutral
            emotions[6] = emotions[6] + face.face_attributes.emotion.sadness
            emotions[7] = emotions[7] + face.face_attributes.emotion.surprise
        print("---------------------------------------------")


def detect_forbidden_scenes(img_list):
    for image_url in img_list:
        image_object = open(os.path.join(images_folder, image_url), "rb")
        computer_vision_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(KEY))
        local_image_features = ["adult"]
        detect_adult_results_local = computer_vision_client.analyze_image_in_stream(image_object, local_image_features)
        print("---------------------------------------------")
        print(f"|For image name: {image_url}|")
        print("---------------------------------------------")
        print("Is adult content: {} with confidence {:.2f}".format(detect_adult_results_local.adult.is_adult_content,
                                                                   detect_adult_results_local.adult.adult_score * 100))
        print("Has racy content: {} with confidence {:.2f}".format(detect_adult_results_local.adult.is_racy_content,
                                                                   detect_adult_results_local.adult.racy_score * 100))

        if detect_adult_results_local.adult.is_adult_content:
            forbidden_content[0] = forbidden_content[0] + 1
        elif detect_adult_results_local.adult.is_racy_content:
            forbidden_content[1] = forbidden_content[1] + 1
        else:
            forbidden_content[2] = forbidden_content[2] + 1
        print("---------------------------------------------")


def images_detection_processing_threads():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for img_name in get_packages("img"):
            executor.submit(detect_forbidden_scenes, img_name)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for img_name in get_packages("img"):
            executor.submit(images_detection_processing, img_name)


def resize_images():
    for name in os.listdir("img"):
        image = Image.open(os.path.join(images_folder, name))
        new_image = image.resize((500, 500))
        new_image.save(os.path.join(images_folder, name), optimize=True, quality=90)


def authenticate_client():
    ta_credential = AzureKeyCredential(KEY)
    text_analytics_client = TextAnalyticsClient(
        endpoint=ENDPOINT,
        credential=ta_credential)
    return text_analytics_client


client = authenticate_client()


def make_all():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for x in os.listdir("text"):
            executor.submit(sentiment_analysis_example, client, x)


# Example function for detecting sentiment in text
def sentiment_analysis_example(client1, name):
    message = open(os.path.join(text_folder, name), "r").read()
    documents = [message]
    response = client1.analyze_sentiment(documents=documents)[0]
    print("---------------------------------------------")
    print(f"|For text name: {name}|")
    print("---------------------------------------------")
    print("Document Sentiment: {}".format(response.sentiment))
    print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
        response.confidence_scores.positive,
        response.confidence_scores.neutral,
        response.confidence_scores.negative,
    ))
    for idx, sentence in enumerate(response.sentences):
        print("Sentence: {}".format(sentence.text))
        print("Sentence {} sentiment: {}".format(idx + 1, sentence.sentiment))
        print("Sentence score:\nPositive={0:.2f}\nNeutral={1:.2f}\nNegative={2:.2f}\n".format(
            sentence.confidence_scores.positive,
            sentence.confidence_scores.neutral,
            sentence.confidence_scores.negative,
        ))
    print("---------------------------------------------")


def create_graph_emotions():
    emotions_dict = {
        'anger': emotions[0],
        'contempt': emotions[1],
        'disgust': emotions[2],
        'fear': emotions[3],
        'happiness': emotions[4],
        'neutral': emotions[5],
        'sadness': emotions[6],
        'surprise': emotions[7]}

    names = list(emotions_dict.keys())
    values = list(emotions_dict.values())

    plt.barh(names, values)
    plt.title('Emotions frecuency')
    plt.ylabel('Emotion type')
    plt.xlabel('Quantity of emotion')
    plt.show()


def create_graph_censored_content():
    plt.barh(['Adult content', 'Racy content', 'All public'], forbidden_content)
    plt.title('Censored content')
    plt.ylabel('Content type')
    plt.xlabel('Censored content frequency')
    plt.show()


if __name__ == '__main__':
    resize_images()
    init_time = time.time()
    sort_files_threads()
    make_all()
    process_videos()
    images_detection_processing_threads()
    print(f"The time of processing is : {time.time() - init_time} seconds")
    create_graph_emotions()
    create_graph_censored_content()
    clean("spliced_videos")
