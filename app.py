from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import json


import model_bow as bow

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('training'))

    return render_template('login.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/training', methods=['POST', 'GET'] )
@login_required
def training():
	if request.method =="POST":
		file = request.files['file_train']
		aspek = ['makanan', 'minuman', 'pelayanan', 'tempat', 'harga']

		#read data
		raw_docs_train = bow.read_data(file)
		print(raw_docs_train)

		#preprocessing data
		processed_data_train = bow.data_preprocessing(raw_docs_train)
		print(processed_data_train)

		#latih model
		# model, x_test, vectorizer, df_akurasi_plot = bow.latih_model(raw_docs_train, aspek)
		
		model_final = {}
		vectorizer_final = {}
		list_df_akurasi = []
		list_confusion_matrix = []
		for i in aspek:
			model, x_test, vectorizer, df_akurasi_plot, y_test, y_pred, gambar_cm = bow.latih_model(raw_docs_train, i)
			list_df_akurasi.append(df_akurasi_plot)
			list_confusion_matrix.append(gambar_cm)     
			model_final[i] = model
			vectorizer_final[i] = vectorizer

		#preprocessing untuk ditampilkan

		# step_preproccess = bow.data_preprocessing(raw_docs_train)
		# print(step_preproccess)

		text_input = str(raw_docs_train["ulasan"][1])
		print(text_input)
		text_case_folding = processed_data_train['lowercase'][1]
		print(text_case_folding)
		text_stopword = processed_data_train['remove_stopwords'][1]
		print(text_stopword)
		text_punct = processed_data_train['punctual'][1]
		print(text_punct)
		text_tokenisasi = processed_data_train['token'][1]
		print(text_tokenisasi)
		text_stemming = processed_data_train['stemmed'][1]
		print(text_stemming)
		text_c_slang = processed_data_train['ubah_slang'][1]
		print(text_c_slang)

		#Menampilkan Plot Training
		list_plot = bow.plot_akurasi(list_df_akurasi)
		plot_eval_makanan = list_plot[0]
		plot_eval_minuman = list_plot[1]
		plot_eval_pelayanan = list_plot[2]
		plot_eval_tempat = list_plot[3]
		plot_eval_harga = list_plot[4]
            
		#Menampilkan Confusion Matrix
		cm_makanan = list_confusion_matrix[0]
		cm_minuman = list_confusion_matrix[1]
		cm_pelayanan = list_confusion_matrix[2]
		cm_tempat = list_confusion_matrix[3]
		cm_harga = list_confusion_matrix[4]
            
		print(model_final)
		print(list_df_akurasi)

		return jsonify({ 'text_input':text_input, 'text_case_folding':text_case_folding, 'text_stopword':text_stopword, 'text_punct':text_punct , 'text_tokenisasi':text_tokenisasi, 'text_stemming' : text_stemming, 'text_c_slang' : text_c_slang , 'plot_eval_makanan' : plot_eval_makanan, 'plot_eval_minuman' : plot_eval_minuman, 'plot_eval_pelayanan' : plot_eval_pelayanan, 'plot_eval_tempat' : plot_eval_tempat, 'plot_eval_harga' : plot_eval_harga, 'cm_makanan' : cm_makanan, 'cm_minuman' : cm_minuman, 'cm_pelayanan' : cm_pelayanan, 'cm_tempat' : cm_tempat, 'cm_harga' : cm_harga })  
	
	return render_template('training.html')

@app.route('/testing', methods=['POST','GET'])
def testing():
		if request.method == "POST":

			aspek = ['makanan', 'minuman', 'pelayanan', 'tempat', 'harga']
			nama_rm = request.form['nama_resto']
			print(nama_rm)
				

			driver = bow.open_chrome()

			search_resto = bow.search_resto(nama_rm, driver)

			redirect_page = bow.redirect_reviewstab(driver)

			scroll_page = bow.scroll_reviewstab(driver)

			df = bow.iteration_allreviews(driver)
			print(df)
			
			preprocess_input = bow.data_preprocessing(df)
			

			list_load_model_svm, list_load_vectorizer = bow.load_model(aspek)

			model_final = {}
			vectorizer_final = {}
			for i in range(len(aspek)):
				model_final[aspek[i]] = list_load_model_svm[i]
				vectorizer_final[aspek[i]] = list_load_vectorizer[i]
                                
			process_input = bow.proses_df_input(preprocess_input, vectorizer_final, model_final, aspek)
            
			print(process_input.head())
                  
			df_process_input = json.loads(process_input[['ulasan', 'makanan', 'minuman', 'pelayanan', 'tempat', 'harga']].to_json(orient="split"))["data"]

                  
			# ulasan_input = json.loads(process_input['ulasan'].to_json(orient="split"))["data"]
			# aspek_makanan_input = json.loads(process_input['makanan'].to_json(orient="split"))["data"]
			# aspek_minuman_input = json.loads(process_input['minuman'].to_json(orient="split"))["data"]
			# aspek_pelayanan_input = json.loads(process_input['pelayanan'].to_json(orient="split"))["data"]
			# aspek_tempat_input = json.loads(process_input['tempat'].to_json(orient="split"))["data"]
			# aspek_harga_input = json.loads(process_input['harga'].to_json(orient="split"))["data"]
			
			#Menampilkan Plot Input
			list_plot_input, total_review = bow.plot_input_rumahmakan(process_input)
			# print(len(list_plot_input))
			
			plot_input_makanan = list_plot_input[0]
			plot_input_minuman = list_plot_input[1]
			plot_input_pelayanan = list_plot_input[2]
			plot_input_tempat = list_plot_input[3]
			plot_input_harga = list_plot_input[4]
			
			jumlah_review = total_review 
            

			return jsonify({'nama_rm':nama_rm , 'plot_input_makanan' : plot_input_makanan,'plot_input_minuman': plot_input_minuman, 'plot_input_pelayanan' : plot_input_pelayanan, 'plot_input_tempat' : plot_input_tempat, 'plot_input_harga' : plot_input_harga, 'jumlah_review' : jumlah_review, 'df_process_input': df_process_input}) 
                
			# 'ulasan_input' : ulasan_input, 'aspek_makanan_input' : aspek_makanan_input, 'aspek_makanan_input' : aspek_minuman_input, 'aspek_minuman_input' : aspek_pelayanan_input, 'aspek_pelayanan_input' : aspek_tempat_input, 'aspek_harga_input' : aspek_harga_input
       
        




if __name__ == "__main__":
	app.run(debug=True)
	#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1
