import repository.gen_queue_repo as gen_queue_repo
import repository.img_document_repo as img_document_repo
import repository.img_upload_repo as img_upload_repo
import repository.img_download_repo as img_download_repo
import ai.workflow.studio as studio
import uuid

def generate_img(queued_img_id: str) -> str:
	queue_entry = gen_queue_repo.get(queued_img_id)
	img_id = queue_entry.image_id

	gen_img_url = studio.workflow(img_id)
	gen_img_id = str(uuid.uuid1())
	img_download_repo.download_img_to_bucket(gen_img_id, gen_img_url)

	file_type = gen_img_url.split(".")[-1]
	img_document_repo.new_img_document(str(gen_img_id), file_type)
	img_upload_repo.add_gen_img_to_img_upload(queue_entry.upload_id, str(gen_img_id))
	gen_queue_repo.del_img_from_queue(queued_img_id)
	return gen_img_id