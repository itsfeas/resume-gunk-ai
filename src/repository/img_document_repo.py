import uuid
from firebase_admin import db
from model.img import Image

QUEUE_DOC_PREFIX = "images_doc/"

# TO-DO: Add error handling
def new_img_document(img_id: uuid.UUID, file_type: str):
    ref = db.reference(QUEUE_DOC_PREFIX)
    ref.child(img_id).set({
        "file_type": file_type,
        "id": img_id,
        "version": 0
    })

def get(img_id: str) -> Image:
    ref = db.reference(QUEUE_DOC_PREFIX)
    obj = ref.child(img_id).get()
    if not obj["id"]:
        raise Exception("Failed to get id of image document from RTDB")
    obj["id"] = uuid.SafeUUID(obj["id"])
    return Image(**obj)