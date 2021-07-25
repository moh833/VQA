from flask import Flask,render_template, request
from QA import generate_answer
import base64
import io
from PIL import Image
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import csv
cred = credentials.Certificate('firebase-sdk.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://shoofli-315622-default-rtdb.firebaseio.com/'
})

ref = db.reference('/')
app = Flask(__name__)


@app.route('/', methods=['get'])
def index():
	return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
	data = request.get_json(force=True)
	image_data = data['image']
	question = data['question']
	question = question[:-1]
	question = question.lower()
	print(question)
	ts = time.time()
	ts =str(int(ts))
	imageName=ts+".jpg"
	imgdata = base64.b64decode(image_data)
	image = Image.open(io.BytesIO(imgdata))
	imagepath="./images/"+ts+".jpg"
	image.save(imagepath)
	image = Image.open(imagepath)
	image.show()

	model_name = 'lstm_qi'
	dataset = 'VQA_1'
	top_5_answers = generate_answer(imagepath, question, dataset, model_name)
	topAnswer = top_5_answers[0]
	print(ts)
	print(top_5_answers)
	data = {'question': question,
                    'imageName': imageName,
                    'topAnswer': topAnswer,
                    'image': image_data,
                    'correct': "2"}
	ref.child("data").push(data)
	return topAnswer

@app.route('/update', methods=['get'])
def test():
	return 'render_template("index.html")'


@app.route('/update', methods=['POST'])
def updatedata():
	data = request.get_json(force=True)
	topAnswer = data['topAnswer']
	ids = data['id']
	data1=   {'correct': "1", 'topAnswer': topAnswer}
	ref = db.reference('data')
	box_ref = ref.child(ids)
	box_ref.update(data1)
	return 'done'

@app.route('/auth', methods=['POST'])
def genCSV():
	data = request.get_json(force=True)
	Pass = data['pass']


	if Pass == 'MahOth159753.':
		print(Pass)
		ref = db.reference('data')
		dta= (ref.get())
		indx = list(dta)
		dectout = []
		for i in indx:
			if dta[i]['correct'] == '1':
				dectout.append({
					'Image': dta[i]['imageName'],
					'question': dta[i]['question'],
					'topAnswer': dta[i]['topAnswer']
				})

		keys = dectout[0].keys()
		with open('dict.csv', 'w') as csv_file:
			dict_writer = csv.DictWriter(csv_file, keys)
			dict_writer.writeheader()
			dict_writer.writerows(dectout)
		return 'done'
	else:
		return 'not correct'








if __name__ == "__main__":
	app.run(host="0.0.0.0",port=5000)
