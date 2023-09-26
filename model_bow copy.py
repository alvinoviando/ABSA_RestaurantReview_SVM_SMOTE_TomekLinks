#Dependencies

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import time
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

from gensim.models import FastText
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

#Function
def read_data(data_file):
    df_raw = pd.read_csv(data_file)

    return df_raw

def remove_stopwords(ulasan):
    ulasan_bersih = []
    
    for kata in ulasan:
        if kata not in list_stopwords_baru:
            ulasan_bersih.append(kata)
    
    return ulasan_bersih

def stemming(ulasan_token):
    ulasan_stem = []
    
    for kata in ulasan_token:
        ulasan_stem.append(stemmer.stem(kata))
        
    return ulasan_stem

def ubah_slang(ulasan_stem):
    kamus_slangword = eval(open("C:\\Users\ASUS\slangwords.txt").read()) # Membuka dictionary slangword
    pattern = re.compile(r'\b( ' + '|'.join(kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
    ulasan_no_slang = []
    for kata in ulasan_stem:
        filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
        ulasan_no_slang.append(filteredSlang)
        
    return ulasan_no_slang

def data_preprocessing(df):
    print()
    df['punctual'] = df['ulasan'].str.replace('[^a-zA-Z]+',' ')
    df['lowercase'] = df['punctual'].str.lower()
    df['token'] = [word_tokenize(i) for i in df['lowercase']]
    df['remove_stopwords'] = [remove_stopwords(i) for i in df['token']]
    df['stemmed'] = [stemming(i) for i in df['remove_stopwords']]
    df['ubah_slang'] = [ubah_slang(i) for i in df['stemmed']]
    
    return df

def latih_model(df, nama_aspek, aspek):
    x = pd.Series([" ".join(i) for i in df['ubah_slang']])
    y = df[aspek]
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    
    model_akurasi = {
    'C' : [],
    'Akurasi' : []
    }

    for c in [0.01, 0.05, 0.25, 0.5, 0.75,  1]:
        svm = LinearSVC(C=c)
        cross_val_score(svm, x_train, y_train, cv=10 )
        svm.fit(x_train, y_train)

        model_akurasi['C'].append(c)
        model_akurasi['Akurasi'].append(accuracy_score(y_test, svm.predict(x_test)))

    df_akurasi = pd.DataFrame(model_akurasi)
    df_akurasi.sort_values(by=['Akurasi'], ascending=False, inplace=True)
    df_akurasi.reset_index(drop=True, inplace=True)
    df_akurasi.to_excel(f'df_akurasi_{nama_aspek}.xlsx',index=False)

    svm = LinearSVC(C=df_akurasi['C'][0])
    svm.fit(x_train, y_train)
    
    print(f'Akurasi tertinggi untuk aspek {aspek} adalah {accuracy_score(y_test, svm.predict(x_test))}')
    
    return svm, x_test, vectorizer

def proses_kalimat_input(kalimat_input, aspek, vectorizer_final, model_final):
    kalimat_input_punctual = kalimat_input.replace('[^a-zA-Z]+',' ')
    kalimat_input_lowercase = kalimat_input_punctual.lower()
    kalimat_input_token = word_tokenize(kalimat_input_lowercase)
    kalimat_input_remove_stopwords = remove_stopwords(kalimat_input_token)
    kalimat_input_stemmed = stemming(kalimat_input_remove_stopwords)
    kalimat_input_ubah_slang = ubah_slang(kalimat_input_stemmed)
    
    kalimat_input_akhir = [" ".join(kalimat_input_ubah_slang)]
    
    
    print("")
    for i in aspek:
        vecs_input = vectorizer_final[i].transform(kalimat_input_akhir)
        prediksi = model_final[i].predict(vecs_input)
        print(f"Memiliki sentimen {prediksi} dalam aspek {i}")
        
def open_chrome():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    #London Victoria & Albert Museum URL
    url = 'https://www.google.co.id/maps/@3.5508653,98.6270636,15z'
    driver.get(url)
    time.sleep(3)
    
    return driver
    
def search_resto(nama_resto, driver):
    # find the search bar element
    search_bar = driver.find_element(By.XPATH, '//*[@id="searchboxinput"]')

    # input the place name or address
    search_bar.send_keys(nama_resto)
    search_bar.send_keys(Keys.RETURN)
    time.sleep(5)

def redirect_reviewstab(driver):
    driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[29]/div/div[2]/button').click()
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div[7]/div[2]/button').click()
    time.sleep(3)
    driver.find_element(By.XPATH,"(//div[@role='menuitemradio'])[3]").click()

def scroll_reviewstab(driver):
    SCROLL_PAUSE_TIME = 5

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    number = 0
    
    while True:
        number = number+1

        # Scroll down to bottom
        ele = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]')
        driver.execute_script('arguments[0].scrollBy(0, 5000);', ele)

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        ele = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]')
        new_height = driver.execute_script("return arguments[0].scrollHeight", ele)

        if number == 10: #NUM ITERATE
            break

        if new_height == last_height:
            break

        last_height = new_height
    
def iteration_allreviews(driver):
    item = driver.find_elements(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div[9]')
    review_list = []

    for i in item:
        button = i.find_elements(By.TAG_NAME, 'button')
        for m in button:
            if m.text == "Lainnya":
                m.click()
        time.sleep(5) 
        review = i.find_elements(By.CLASS_NAME, "wiI7pd")
        
        for ulasan in review:
            review_list.append(ulasan.text)
        
    review_list = [i for i in review_list if len(i) > 10]
            
    df_test_ulasan = pd.DataFrame({'ulasan': review_list})

    
    return df_test_ulasan

def proses_df_input(df, vectorizer_final, model_final, aspek):
    dict_predict = {}
    for i in aspek:
        dict_predict[i] = []
        
    for i in df['ubah_slang']:
        kalimat_input_akhir = [" ".join(i)]
        
        for i in dict_predict.keys():
            vecs_input = vectorizer_final[i].transform(kalimat_input_akhir)
            prediksi = model_final[i].predict(vecs_input)
            prediksi = str(list(prediksi))
            prediksi = prediksi.replace("'",'')
            prediksi = prediksi.replace('[','')
            prediksi = prediksi.replace(']','') 
            dict_predict[i].append(prediksi)
    
    for i in dict_predict.keys():
        df[i] = dict_predict[i]  # menambah kolom setiap aspek ke df, dengan isinya list setiap aspek dalam dict
    
    return df

def summary(df, aspek):
    
    dict_sentimen_final = {}
    for i in aspek:
        dict_sentimen_final[f'sentimen_{i}_final'] = df[i].mode()[0]
        
#     sentimen_makanan_final = df['makanan'].mode()[0]
#     sentimen_minuman_final = df['minuman'].mode()[0]
#     sentimen_pelayanan_final = df['pelayanan'].mode()[0]
#     sentimen_tempat_final = df['tempat'].mode()[0]
#     sentimen_harga_final = df['harga'].mode()[0]

    return dict_sentimen_final

def tes_prediksi():
    nama_resto = 'Bakso Aci Juara'
    driver = open_chrome()
    search_resto(nama_resto, driver)
    redirect_reviewstab(driver)
    scroll_reviewstab(driver)
    df_test_ulasan = iteration_allreviews(driver)
    df_test_ulasan

    df_testing = data_preprocessing(df_test_ulasan)
    df_testing = proses_df_input(df_testing)
    dict_sentimen = summary(df_testing)
    print(dict_sentimen)

#System

df = pd.read_csv('dataset_final.csv')

stop_words = set(stopwords.words('indonesian'))
list_stopwords_baru = stop_words
# list_neg = ['tidak', 'tidaklah','lawan','anti', 'belum', 'belom', 'tdk', 'jangan', 'gak', 'enggak', 'bukan', 'sulit', 'tak', 'sblm']
# list_stopwords_baru = list(set(stop_words) - set(list_neg))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

df = data_preprocessing(df)

# aspek = ['makanan', 'minuman', 'pelayanan', 'tempat', 'harga']
# model_final = {}
# vectorizer_final = {}

# for i in aspek:
#     model, x_test, vectorizer = latih_model(df, i)
#     model_final[i] = model
#     vectorizer_final[i] = vectorizer

