import uuid
from firebase_admin import db
from model.img import ImageUpload

DOC_PREFIX = "img_upload/"

# TO-DO: Add error handling
def new_img_upload_document() -> dict:
    ref = db.reference(DOC_PREFIX)
    get = ref.get()
    if not get:
        return None
    # for g in get:
    #     print(ref.child(g).get().items())
    doc_id = next(iter(get))
    return ref.child(doc_id).get()
    # for doc in docs:
    #     return doc.to_dict()

def add_gen_img_to_img_upload(img_upload_id: str, gen_img_id: str) -> None:
    ref = db.reference(DOC_PREFIX)
    ref.child(img_upload_id).child("generated_images").push(gen_img_id)

# def get(img_id: str) -> ImageUpload:
#     ref = db.reference(DOC_PREFIX)
#     obj = ref.child(img_id).get()

#     ref = db.reference(DOC_PREFIX)
#     obj = ref.child(img_id).get()

#     # converting string to uuids
#     _convert_to_uuid(obj, "id")
#     _convert_to_uuid(obj, "image")
#     field = "generated_images"
#     if obj[field]:
#         for i in range(len(obj[field])):
#             obj[field][i] = uuid.SafeUUID(obj[field][i])
#     return ImageUpload(**obj)

# def _convert_to_uuid(obj: dict, field_name: str):
#     if not obj[field_name]:
#         raise Exception("Failed to get field, {field}, of image_upload doc from RTDB".format(field=field_name))
#     obj[field_name] = uuid.SafeUUID(obj[field_name])