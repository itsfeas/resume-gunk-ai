import datetime
import os
from firebase_admin import storage
import requests

def get_url_from_img_id(img_id: str) -> str:
    bucket_name = os.environ.get("IMG_STORAGE_BUCKET")
    bucket = storage.bucket(bucket_name)
    blob = bucket.get_blob(img_id)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(days=1),
        method="GET",
    )
    # print("generated url", url)
    return url

def download_img_to_bucket(img_id: str, img_url: str):
    bucket_name = os.environ.get("IMG_STORAGE_BUCKET")
    bucket = storage.bucket(bucket_name)

    response = requests.get(img_url)
    if response.status_code == 200:
        blob = bucket.blob(img_id)
        blob.upload_from_string(response.content)
    else:
        raise Exception("Failed to download image from URL")