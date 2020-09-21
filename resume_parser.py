import PyPDF2
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
import streamlit as st
import pdfminer
from pdfminer.high_level import extract_text
import re
import matplotlib.pyplot as plt
import io
st.set_option('deprecation.showfileUploaderEncoding', False)
# Get all type of resumes and its distribution

def get_url(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, text)
    return [i[0] for i in url]

def get_exp (text):
    regex = r"\d*\.?\d+"
    exp = re.findall(regex,text)
    return [i[0] for i in exp]

st.sidebar.header('Resume part')
location = None
uploaded_file = st.sidebar.file_uploader("choose resume", type="pdf")
if uploaded_file is not None:
    file = extract_text(uploaded_file)
    file = file.replace("\n", "")
    file = file.lower()
    urls = get_url(file)
    experience = get_exp(file)
    # st.header("Resume contents")
    # st.header("Extracted URL's")
    # st.write(urls)
    # st.header("Extracted years")
    # st.write (experience)
    # st.write(file)
    location = None


location = st.sidebar.text_input('Enter the folder name to select multiple cvs')
if location is not None:
    all_cv = [file for file in listdir(location)]
    selected_cvs = st.sidebar.multiselect('select cvs',all_cv,all_cv)
    uploaded_file  = None

mypath = location #Resume's folder
onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
st.header("Resume Parsing")
st.write(f"we have {len(onlyfiles)} CV's to parse")

def get_text(file):
    return extract_text(file)


def create_profile(file):
    text = get_text(file)
    #     text = pdfextract(file)
    text = str(text)
    text = text.replace("\n", "")
    text = text.lower()
    # below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv(r"D:/Projects/NWOW/Resume_parser/keywords_template.csv", encoding='latin-1')
    ML_words = [nlp(text) for text in keyword_dict['machine_learning'].dropna(axis=0)]
    DL_words = [nlp(text) for text in keyword_dict['deep_learning'].dropna(axis=0)]
    NLP_words = [nlp(text) for text in keyword_dict['nlp'].dropna(axis=0)]
    Data_Engineering_words = [nlp(text) for text in keyword_dict['data_engineering'].dropna(axis=0)]
    Big_Data_words = [nlp(text) for text in keyword_dict['big_data'].dropna(axis=0)]
    stats_words = [nlp(text) for text in keyword_dict['statistics'].dropna(axis=0)]
    python_words = [nlp(text) for text in keyword_dict['python'].dropna(axis=0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('ML', None, *ML_words)
    matcher.add('DL', None, *DL_words)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('DE', None, *Data_Engineering_words)
    matcher.add('BD', None, *Big_Data_words)
    matcher.add('Stats', None, *stats_words)
    matcher.add('Python', None, *python_words)

    doc = nlp(text)

    d = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start: end]  # get the matched slice of the doc
        d.append((rule_id, span.text))
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i, j in Counter(d).items())

    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords), names=['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ', 1).tolist(), columns=['Subject', 'Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(', 1).tolist(), columns=['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'], df2['Keyword'], df2['Count']], axis=1)
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]

    name = filename.split(' ')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2), names=['Candidate Name'])

    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis=1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace=True)

    return (dataf)

final_database=pd.DataFrame(dtype=int)
i = 0
while i < len(onlyfiles):
    file = onlyfiles[i]
    dat = create_profile(file)
    final_database = final_database.append(dat)
    i +=1
    #print(final_database)

final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
final_database2.reset_index(inplace = True)
final_database2.fillna(int(0),inplace=True)
new_data = final_database2.iloc[:,1:]
new_data.index = final_database2['Candidate Name']
#execute the below line if you want to see the candidate profile in a csv format
sample2 = new_data.to_excel(r"D:/Projects/NWOW/Resume_parser/candidate_skill_matrix_19sep.xlsx")
# st.dataframe(new_data.style.highlight_max(axis=0))
st.dataframe(new_data,700,900)
plt.rcParams.update({'font.size': 40})
ax = new_data.plot.barh(title="Resume keywords by category", legend=True, figsize=(105,55), stacked=True)
labels = []
for j in new_data.columns:
    for i in new_data.index:
        label = str(j)+": " + str(new_data.loc[i][j])
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center')
# plt.savefig('candidate_skills_19sep.jpg', dpi=300, transparent=True)
# st.pyplot(ax)
# st.pyplot(height=800,width=800)

