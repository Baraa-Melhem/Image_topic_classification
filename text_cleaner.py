import warnings
warnings.filterwarnings("ignore")


import neattext as nt
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

# text prepocessing & cleaning
class TextCleaner:
    def __init__(self, text):
        self.text = text
    
    def clean_text(self):
        docx = nt.TextFrame(text=self.text)
        docx.text
        cleaned_text = docx.normalize(level='deep')
        cleaned_text = docx.remove_puncts()
        cleaned_text = docx.remove_stopwords()
        cleaned_text = docx.remove_html_tags()
        cleaned_text = docx.remove_special_characters()
        cleaned_text = docx.remove_emojis()
        cleaned_text = docx.fix_contractions()
        return cleaned_text
    
    def lemmatize_text(self, cleaned_text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(cleaned_text)
        postex = []
        
        for token in doc:
            wordtext = token.text
            poswrd = spacy.explain(token.pos_)

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
        
        return ",".join(postex)
    
    def lemmatize_with_wordnet(self, pos_tagged_text):
        lemmatizer = WordNetLemmatizer()
        lemmatized_text = ""
        tokens = [pair.strip("()").split("),") for pair in pos_tagged_text.split("),(")]
        for word_pos in tokens:
            if len(word_pos) == 1:
                word, pos = word_pos[0].rstrip(")").rsplit(")(")
            else:
                word, pos = word_pos
            
            if pos == "dt" or pos == "cc" or pos == "prep" or pos == "intj":
                pos = wordnet.NOUN
            
            lemmatized_text = lemmatizer.lemmatize(word, pos=pos) + " " + lemmatized_text
            text = lemmatized_text
            filtered_text = re.sub(r'\bhttps\w*\b', '', text) 
            lemmatized_text= filtered_text
        return lemmatized_text