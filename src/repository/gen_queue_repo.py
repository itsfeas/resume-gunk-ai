from firebase_admin import db
from model.queue import QueuedImage

QUEUE_DOC_PREFIX = "gen_queue/"

# TO-DO: Add error handling
def get_one_queued_img() -> QueuedImage:
    ref = db.reference(QUEUE_DOC_PREFIX)
    get = ref.get()
    if not get:
        return None
    # for g in get:
    #     print(ref.child(g).get().items())
    doc_id = next(iter(get))
    obj = ref.child(doc_id).get()
    return QueuedImage(**obj)

def get(img_id: str) -> QueuedImage:
    ref = db.reference(QUEUE_DOC_PREFIX)
    obj = ref.child(img_id).get()
    print("queued_img", obj)
    return QueuedImage(**obj)

def del_img_from_queue(img_id: str):
    ref = db.reference(QUEUE_DOC_PREFIX)
    ref.child(img_id).delete()