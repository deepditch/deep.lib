import requests
import json
import logging
import os
import xml.etree.cElementTree as ET
import urllib
from PIL import Image
from datetime import datetime

DATASET_FOLDER = "C:/fastai/courses/dl2/data/road_damage_dataset/Data"
LOGFILE = DATASET_FOLDER + '/log.log'
TIME_LOG = DATASET_FOLDER + '/time_log.txt'

logging.basicConfig(filename=LOGFILE, level=logging.DEBUG)

STORAGE_SERVER_URL = "http://209.126.30.247"
API_URL = "http://209.126.30.247/api"
EMAIL = "machine@learning.com"
PASSWORD = "mlmodelupload"

TODAY = datetime.now().strftime('%Y-%m-%d')

def get_last_download_date():
    if not os.path.exists(TIME_LOG):
        return "1900-1-1"

    with open(TIME_LOG, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        return last_line

LAST_DATE = get_last_download_date()

def login():
    url = API_URL + "/login"  
    headers = {"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"}
    params = {"email": EMAIL, "password": PASSWORD}
    r = requests.post(url=url, data=json.dumps(params), headers=headers)

    if(r.status_code != 200):
        logging.error("Login Failure")
        raise Exception("Login Failure")

    return r.json()['access_token']

TOKEN = login()

def get_damages():
    url = API_URL + "/road-damage/verified-images"
    headers = {"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest", "Authorization": "Bearer " + TOKEN}
    params = {"after": LAST_DATE}
    r = requests.get(url=url, data=json.dumps(params), headers=headers)
    if(r.status_code != 200):
        logging.error("Failed to get verified images")
        raise Exception("Failed to get verified images")
    
    logging.info(f"Getting images verified since {LAST_DATE}. Today's date is {TODAY}")
    return r.json()

def download_image(image, dir):
    image_url = STORAGE_SERVER_URL + image['url']
    image_file_name = image['url'].split("/")[-1]
    image_save_path = DATASET_FOLDER + f"/{dir}" + "/JPEGImages/" + image_file_name
    annotation_save_path = DATASET_FOLDER + f"/{dir}" + "/Annotations/" + os.path.splitext(image_file_name)[0]+'.xml'

    # download the image
    urllib.request.urlretrieve(image_url, image_save_path)

    im=Image.open(image_save_path)
    width, height = im.size

    root = ET.Element("root")
    doc = ET.SubElement(root, "annotation")

    ET.SubElement(doc, "folder").text = dir
    ET.SubElement(doc, "filename").text = image_file_name

    size = ET.SubElement(doc, "size") 

    ET.SubElement(size, "width").text = str(width) 
    ET.SubElement(size, "height").text = str(height) 

    damages = image['types'] if not isinstance(image['types'], str) else [image['types']]
    for damage in damages:
        obj = ET.SubElement(doc, "object")
        ET.SubElement(obj, "name").text = damage

    tree = ET.ElementTree(root)
    tree.write(annotation_save_path)

def download_images(image_array):
    folder = DATASET_FOLDER + f"/{TODAY}"
    if not os.path.exists(folder):
        os.mkdir(folder)
        logging.info(f"Creating directory {folder}")

    JPEG_DIRECTORY = folder + "/JPEGImages"
    if not os.path.exists(JPEG_DIRECTORY):
        os.mkdir(JPEG_DIRECTORY)
        logging.info(f"Creating directory {JPEG_DIRECTORY}")

    ANNOTATIONS_DIRECTORY = folder + "/Annotations"
    if not os.path.exists(ANNOTATIONS_DIRECTORY):
        os.mkdir(ANNOTATIONS_DIRECTORY)
        logging.info(f"Creating directory {ANNOTATIONS_DIRECTORY}")

    for image in image_array.values():
        try:
            download_image(image, TODAY)
        except Exception as e:
            logging.error(f"Error downloading and saving {STORAGE_SERVER_URL + image['url']}: {e}", exc_info=True)
            continue

    with open(TIME_LOG, 'w') as f:
        f.write(f'{TODAY} \n')

download_images(get_damages())
 