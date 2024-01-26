from dataclasses import dataclass
import uuid

@dataclass
class ImageUpload:
	id: uuid.UUID
	image: uuid.UUID
	generated_images: list[uuid.UUID]
	created_at: str
	updated_at: str
	version: int


@dataclass
class Image:
	id: uuid.UUID
	file_type: str
	version: int