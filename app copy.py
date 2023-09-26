from flask import Flask, render_template, request, jsonify

import model as ml
# import testing as ts
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/training', methods=['POST','GET'])
def training():
	if request.method=="POST":
		file = request.files['file_train']
		#read data
		raw_docs_train, raw_docs_val, y_train, y_val = ml.read_data(file)

		#preprocessing
		processed_docs_train = ml.preprocessing(raw_docs_train)
		processed_docs_val= ml.preprocessing(raw_docs_val)

		#tokenization
		word_index, word_seq_train, word_seq_val = ml.tokenization_training(processed_docs_train, processed_docs_val)

		#model
		loss_plot, acc_plot, loss_train, accuracy_train, loss_val, accuracy_val = ml.model_building(word_index, word_seq_train, word_seq_val, y_train, y_val)

		#step preproccessing untuk interface
		step_preproccess = ml.step_preproccess(raw_docs_train)

		text_input = str(raw_docs_train[2])
		text_case_folding = raw_docs_train[2].lower()
		text_stopword = step_preproccess[0].lower()
		text_punct = step_preproccess[1].lower()
		text_tokenisasi = str(word_seq_train[2])
	
		#accuracy = tr.predict(model, word_seq_train, y_train)
	return jsonify({'loss_plot':loss_plot,'acc_plot':acc_plot,'loss_train':loss_train, 'accuracy_train':accuracy_train,'loss_val':loss_val, 'accuracy_val':accuracy_val, 'text_input':text_input, 'text_case_folding':text_case_folding, 'text_stopword':text_stopword, 'text_punct':text_punct, 'text_tokenisasi':text_tokenisasi})
		# return render_template('home.html', message=accuracy, loss_plot=loss_plot, acc_plot=acc_plot)

@app.route('/testing', methods=['POST','GET'])
def testing():
	if request.method=="POST":
		file = request.files['file_test']
		accuracy, output_pred = ml.testing(file)

		data_output = []
		for i in range (len(output_pred)):
			data_output.append({
					'id': str(i+1),
                    'title': str(output_pred['title'][i]),
                    'label_score': str(output_pred['label_score'][i]),
                    'prediction': str(output_pred['pred'][i]),
                })
	return jsonify({'accuracy':accuracy, 'data_output':data_output})
	#return jsonify({'accuracy':accuracy})

@app.route('/predict', methods=['POST','GET'])
def predict():
	if request.method=="POST":
		data_pred = request.form['data_pred']
		hasil_pred = ml.predict(data_pred)
		
	return jsonify({'hasil_pred':hasil_pred})
	#return jsonify({'accuracy':accuracy})


if __name__ == "__main__":
	app.run(debug=True)
	#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1