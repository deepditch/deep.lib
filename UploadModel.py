import requests
import json
import logging
import os
import xml.etree.cElementTree as ET
import urllib
from PIL import Image
from datetime import datetime
import sys

API_URL = "https://www.deepditch.com/api"

def login():
    url = API_URL + "/login"  
    headers = {"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"}
    params = {"email": EMAIL, "password": PASSWORD}
    r = requests.post(url=url, data=json.dumps(params), headers=headers)

    if(r.status_code != 200):
        logging.error("Login Failure")
        raise Exception("Login Failure")

    return r.json()['access_token']

def main(args):
    TOKEN = login(args.email, args.password)
    url = API_URL + "/machine-learning/upload-model"  
    headers = {"X-Requested-With": "XMLHttpRequest", "Authorization": "Bearer " + TOKEN}
    r = requests.post(url, headers=headers, files=dict(model=open(args.model_file,'rb')))
    print(r.status_code) 
    if r.status_code == 201:
        print(f"Successfully uploaded {args.model_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file', metavar='Model File')
    
    keyword_args = ['--email', '--password', '--api_address']

    for arg in keyword_args:
        parser.add_argument(arg)

    args = parser.parse_args()

    if args.api_address is not None:
        API_URL = args.api_address

    main(args)