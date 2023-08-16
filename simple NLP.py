import pandas as pd 
from tqdm import tqdm
#reading data
data = pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\Data\topic_classification_data.csv",encoding='utf-8')

# simple data cleaning before starting in NLP
data.describe(include="all")
data.drop(columns=['Unnamed: 0'], inplace=True)
data.drop(columns=['Unnamed: 1'], inplace=True)

data.dropna(axis=0, inplace=True)

data.isna().sum()
data["label"].value_counts()
 


### NLP ###

#!pip install neattext


# text cleaning 




# text cleaning libs
import neattext as nt
import nltk
#nltk.download()  # in case you did not download it here is the steps :remove the "#" before "nltk.download()" ,then choose : 1>>d 2>>l 3>>all

total_rows = 136798 # this is the number of raws make sure to put the right number
cind=0# the index of the column that you want to apply NLP on
with tqdm(total=total_rows) as pbar:
    for i in range(len(data["content"])):
        mytext=data.iloc[i,cind] 
        docx = nt.TextFrame(text=mytext)
        docx.text 
        data.iloc[i,cind]=docx.normalize(level='deep')
        data.iloc[i,cind]=docx.remove_puncts()
        data.iloc[i,cind]=docx.remove_stopwords()
        data.iloc[i,cind]=docx.remove_html_tags()
        data.iloc[i,cind]=docx.remove_special_characters()
        data.iloc[i,cind]=docx.remove_emojis()
        data.iloc[i,cind]=docx.fix_contractions()
        data.iloc[i,cind]
        
        pbar.update(1)


print("Data cleaning completed.")


# POS tagging
#it could be :
# [noun
# verb
# adjective
# adverb
# pronoun
# determiner
# conjunction
# preposition
# interjection
# common noun
# proper noun
# mass noun
# count noun

dataaf=data

nwtxt=[]

import spacy

nlp = spacy.load('en_core_web_sm')
newcleantext = []

with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["content"])):
        doc1 = nlp(dataaf.iloc[i, cind])
        postex = []
        
        for token in doc1:
            wordtext = token.text
            poswrd = spacy.explain(token.pos_)

            # Map POS to single characters
            if poswrd == "verb":
                poswrd = "v"
            elif poswrd == "noun":
                poswrd = "n"
            elif poswrd == "adjective":
                poswrd = "a"
            elif poswrd == "adverb":
                poswrd = "r"
            elif poswrd == "pronoun":
                poswrd = "n"
            elif poswrd == "determiner":
                poswrd = "dt"
            elif poswrd == "conjunction":
                poswrd = "cc"
            elif poswrd == "preposition":
                poswrd = "prep"
            elif poswrd == "interjection":
                poswrd = "intj"
            elif poswrd == "common noun":
                poswrd = "n"
            elif poswrd == "proper noun":
                poswrd = "n"
            elif poswrd == "mass noun":
                poswrd = "n"
            elif poswrd == "count noun":
                poswrd = "n"
            else:
                poswrd = "n"

            postex.append(f"({wordtext})({poswrd})")

        newcleantext.append(",".join(postex))
        pbar.update(1)

lemmtext = {"cleantext2": newcleantext}
newtextline = pd.DataFrame(lemmtext)
print("completed..")

dataaf=pd.concat([dataaf,newtextline],axis=1)


####
dataaf.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\New clean data\1CLEAN.csv",index=False)


dataaf=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\New clean data\1CLEAN.csv")



#lemmatization
# pos parameter
# "n" for nouns
# "v" for verbs
# "a" for adjectives
# "r" for adverbs
# "s" for satellite adjectives
# Determiner	dt
# Conjunction	cc
# Preposition	prep
# Interjection	intj
# Noun	n
# pos=wordnet.NOUN


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm

newcleantext = []
lemmatizer = WordNetLemmatizer()
dataaf.dropna(axis=0, inplace=True)

# after any 'dataaf.dropna(axis=0, inplace=True)' you need to make sure to enter the right total raws

total_rows = 136619
cind=2 # the index of the column that you want to apply NLP on 

with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["cleantext2"])):
        textsve = ""
        text_in_data = dataaf.iloc[i, cind]
        tokens = [pair.strip("()").split("),") for pair in text_in_data.split("),(")]
        for word_pos in tokens:
            if len(word_pos) == 1:
                
                word, pos = word_pos[0].rstrip(")").rsplit(")(")
            else:
                word, pos = word_pos
                
            if pos == "dt" or pos == "cc" or pos == "prep" or pos == "intj":
                pos = wordnet.NOUN
            
            textsve = lemmatizer.lemmatize(word, pos=pos) +" "+ textsve

        newcleantext.append(textsve)
        pbar.update(1)

lemmtext = {"cleantext": newcleantext}
lemmtext = pd.DataFrame(lemmtext)
print("done")

dataaf=dataaf.drop("cleantext2",axis=1)
dataaf=dataaf.drop("content",axis=1)

dataaf=pd.concat([dataaf,lemmtext],axis=1)

dataaf.isna().sum()
dataaf.dropna(axis=0, inplace=True)


dataaf.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\New clean data\2CLEAN.csv",index=False)


dataaf=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\New clean data\2CLEAN.csv")

