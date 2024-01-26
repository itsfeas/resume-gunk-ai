import os
import firebase_admin

def init_client() -> firebase_admin.App:
    # Fetch the service account key JSON file contents
    credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    url = os.environ["FB_RTDB_URL"]

    cred_obj = firebase_admin.credentials.Certificate(credentials)
    return firebase_admin.initialize_app(cred_obj, {
        'databaseURL': url
    })