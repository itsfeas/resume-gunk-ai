import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

def init_client() -> storage.bucket:
    credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    bucket = os.environ["IMG_STORAGE_BUCKET"]
    
    cred_obj = firebase_admin.credentials.Certificate(credentials)
    firebase_admin.initialize_app(cred_obj, {
        'storageBucket': f'{bucket}.appspot.com'
    })

    return storage.bucket()