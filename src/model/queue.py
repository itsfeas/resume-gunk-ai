
from dataclasses import dataclass
import uuid

@dataclass
class QueuedImage:
	upload_id: uuid.UUID
	image_id: uuid.UUID