import requests
import json
import logging
import os
import xml.etree.cElementTree as ET
import urllib
from PIL import Image
from datetime import datetime
import argparse

STORAGE_SERVER_URL = "https://www.deepditch.com/"
API_URL = "https://www.deepditch.com/api"

def get_last_download_date(logfile):
    if not os.path.exists(logfile):
        return "1900-1-1"

    with open(logfile, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        return last_line

def login(email, password):
    url = API_URL + "/login"  
    headers = {"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"}
    params = {"email": email, "password": password}
    r = requests.post(url=url, data=json.dumps(params), headers=headers)

    if(r.status_code != 200):
        logging.error("Login Failure")
        print(r)
        raise Exception("Login Failure")

    return r.json()['access_token']

def get_damages(token, last_date, today):
    url = API_URL + "/road-damage/verified-images"
    headers = {"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest", "Authorization": "Bearer " + token}
    params = {"after": last_date}
    r = requests.get(url=url, data=json.dumps(params), headers=headers)

    if(r.status_code != 200):
        logging.error("Failed to get verified images")
        raise Exception("Failed to get verified images")
    
    logging.info(f"Getting images verified since {last_date}. Today's date is {today}")

    return r.json()

def download_image(image, data_dir, dir):
    image_url = STORAGE_SERVER_URL + image['url']
    image_file_name = image['url'].split("/")[-1]
    image_save_path = data_dir + f"/{dir}" + "/JPEGImages/" + image_file_name
    annotation_save_path = data_dir + f"/{dir}" + "/Annotations/" + os.path.splitext(image_file_name)[0]+'.xml'

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

def download_images(data_dir, image_array, today, time_log):
    if image_array == []: 
        print("No new images")
        return

    folder = data_dir + f"/{today}"
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
            download_image(image, data_dir, today)
        except Exception as e:
            logging.error(f"Error downloading and saving {STORAGE_SERVER_URL + image['url']}: {e}", exc_info=True)
            continue

    with open(time_log, 'w') as f:
        f.write(f'{today} \n')

def main(args):
    data_dir = args.data_dir
    logfile = data_dir + '/log.log'
    time_log = data_dir + '/time_log.txt'

    logging.basicConfig(filename=logfile, level=logging.DEBUG)

    last_date = get_last_download_date(time_log)
    today = datetime.now().strftime('%Y-%m-%d')

    token = login(args.email, args.password)

    images = get_damages(token, last_date, today)
 
    download_images(data_dir, images, today, time_log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', metavar='DataDirectory')

    keyword_args = ['--email', '--password', '--api_address', '--storage_address']

    for arg in keyword_args:
        parser.add_argument(arg)

    args = parser.parse_args()

    if args.api_address is not None:
        API_URL = args.api_address
    
    if args.storage_address is not None:
        STORAGE_SERVER_URL = args.storage_address

    main(args)