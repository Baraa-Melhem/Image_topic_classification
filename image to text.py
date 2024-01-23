#to read any text in image :
    
import pytesseract
import cv2
import numpy as np

from keras.models import load_model
import pickle
# recall the model
model2 = load_model(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\model\img2text_classi_model.h5")
3
file_path = r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\model/tokenizer.pickle'
with open(file_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
    

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe" # must put the path of this package that in your device   

#image that you want to convert it into text
img = cv2.imread(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\image exp\site-ad-bb-ball-2022_orig.jpg")
txt = pytesseract.image_to_string(img)
print("\n")
print("text:",txt)



from keras.preprocessing.sequence import pad_sequences 

from text_cleaner import TextCleaner# it's a class that I made so I can make it easier to clean text
mytext = txt
cleaner = TextCleaner(mytext)
cleaned_text = cleaner.clean_text()
pos_tagged_text = cleaner.lemmatize_text(cleaned_text)
lemmatized_text = cleaner.lemmatize_with_wordnet(pos_tagged_text)
print("\n","clean text:",lemmatized_text,"\n")



sequences = loaded_tokenizer.texts_to_sequences([lemmatized_text])
max_sequence_length = 20  
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

predictions = model2.predict(padded_sequences)
predictedind= np.argmax(predictions[0])# index of the highest probability

if predictedind==0:
    predictedind="Health"
elif predictedind==1:
    predictedind="Politics"
elif predictedind==2:
    predictedind="Emotion"
elif predictedind==3:
    predictedind="financial"
elif predictedind==4:
    predictedind="other topic , might be sport or science"



print("text topic :", predictedind)
