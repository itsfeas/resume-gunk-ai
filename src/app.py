import os
from os.path import join, dirname
import service.img_generation_service as img_generation_service
from flask import Flask
import db.rtdb.client as rtdb
from dotenv import load_dotenv

# Load environment variables from the .env file
dotenv_path = join(dirname(__file__), 'env/.env')
load_dotenv(dotenv_path)

# Load Firebase
rtdb.init_client()

# Create Flask app
app = Flask(__name__)


@app.route("/api/v1/generate/<string:queued_img_id>", methods=["GET"])
def generate(queued_img_id):
	print("queued_img_id", queued_img_id)
	gen_img_id = img_generation_service.generate_img(queued_img_id)
	return {
		"status": 200,
		"msg": gen_img_id,
	}

if __name__ == "__main__":
	app.run()
