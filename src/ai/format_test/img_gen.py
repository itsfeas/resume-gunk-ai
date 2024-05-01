from os import walk
import fitz

def list_files(mypath):
	f = []
	for (dirpath, dirnames, filenames) in walk(mypath):
		f.extend(filenames)
		break
	return(f)

def dataset_gen_img(path: str):
	for i, p in enumerate(list_files(path)):
		if not p.startswith("train_"):
			continue
		print(path+p)
		doc = fitz.open(path+p)
		for i in range(doc.page_count):
			img = doc.load_page(i).get_pixmap()
			img_name = p.replace(".pdf", f"_{i}.png")
			img.save(path+"img/"+img_name)

if __name__ == "__main__":
	dataset_gen_img("./dataset/")
		