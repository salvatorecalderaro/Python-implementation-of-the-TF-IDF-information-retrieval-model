"""
IMPLEMENTAZIONE DI TF-IDF
@author: Salvatore Calderaro
"""

import numpy as np
from math import log2
from scipy.spatial import distance

"""
Estraggo le frasi dal testo.
""" 
def find_phrases(text):
    text=text.lower()
    phrases=text.split(".")
    del phrases[-1]
    return phrases

"""
Creo il vocabolario
"""
def create_vocabulary(text):
    text=text.lower()
    words=text.split(" ")
    voc=list(set(words))
    value_dict = []
    for i in range(len(voc)):
        value_dict.append((i, voc[i]))
    vocab = dict((y, x) for x, y in value_dict)
    return voc,vocab


"""
Indice invertito 
Necessario per calcolare n_i in IDF 
"""

def inverted_index(phrases,vocab):
    iv={}
    for i in range (len(vocab)):
        aus=[]
        for j in range (len(phrases)):
            if vocab[i] in phrases[j]:
                aus.append(j)
        iv[vocab[i]]=aus
    return iv


"""
Calcolo di TF 
"""
def compute_tf(word, doc):
    l =doc.split(" ")
    f=l.count(word)
    if (f >0):
        tf=(1+ log2(f))
    else:
        tf=0
    return tf

"""
Calcolo di IDF
"""
def compute_idf(N,word,iv):
    n_i=len(iv[word])
    if(n_i>0):
        idf=log2(N/n_i)
    else:
        idf=0
    return idf 

"""
Calcolo di TF-IDF
"""
def compute_tf_idf(doc,voc,iv):
    N=len(doc)
    tf_idf=np.zeros((len(voc),len(doc)))
    for i in range(len(voc)):
        for j in range(len(doc)):
            tf=compute_tf(voc[i],doc[j])
            idf=compute_idf(N,voc[i],iv)
            tf_idf[i][j]=tf*idf
    return tf_idf.T

"""
Stampo il documento e la sua relativa rappresentazione vettoriale TF-IDF 
"""
def print_tf_idf(tf_idf,doc):
    for i in range(len(doc)):
        print("Documento %d: %s" %(i,doc[i]))
        print(*tf_idf[i,:])
        print("*************************************************************************************")

# Calcolo TF-IDF della query
def compute_tf_idf_query(N,query,voc,iv):
    query=query.lower()
    tf_idf=np.zeros(len(voc))
    for i in range(len(voc)):
        tf=compute_tf(voc[i],query)
        idf=compute_idf(N,voc[i],iv)
        tf_idf[i]=tf*idf
    return tf_idf

# Calcolo della distanza coseno tra query ed ogni documento
def cosine_distance(qv,tf_idf):
    values=[]
    for i in range(tf_idf.shape[0]):
        v=distance.cosine(qv,tf_idf[i,:])
        values.append((i,v))
    values=sorted(values, key=lambda tup: (tup[1]))
    return values

def print_cd(values):
    for v in values:
        print(v)

if __name__ == '__main__':
    text1="The Philippine Ambassador to Lebanon, Bernardita Catalla, has died from complications of Covid-19, the embassy said in a statement on Thursday. Catalla, a career diplomat for 27 years, died in hospital in Beirut. Months before her death, she spearheaded the evacuation of nearly 2,000 Filipinos, mostly female domestic workers, from crisis-ridden Lebanon. Lebanon has been in the throes of a financial and political crisis since October 2019, and many African and Asian migrant workers in the country reported a large drop in earnings and the withholding of salaries. The Philippine embassy’s evacuation program for Filipinos, many of whom were undocumented, received praise from humanitarian workers and international rights groups. CNN interviewed Catalla twice about her repatriation program since December. “Those who feel that they have to go back to the Philippines do so because they have nothing here anymore. Some have become homeless,” Catalla told CNN in February. Women clamored for selfies with the ambassador as they waited for the bus to take them to the airport. Asked what she thought about the praise being heaped on the embassy for their evacuation program, Catalla responded: “We’re just doing our jobs.” Philippine Foreign Affairs Secretary Teodro Locsin Jr tweeted that before Catalla died he “extended her a great job in a difficult post. I promised her Paris so she’d hang on.” “But she just laughed, ‘Now I must learn French,’” wrote Locsin. “Ambassador Bernie Catalla’s remains will be received with an honor guard and I am putting forward a nomination for Gawad Mabini and Sikatuna,” said Locsin, referring to an honor conferred on Filipinos for distinguished foreign service.  “Not that she needs more honor than the profound regret and mourning of a grateful service, government, and I hope nation.” Spain deaths pass 10,000 after highest single-day increase so far. At least 10,003 people have now died after testing positive for coronavirus in Spain, according to Health Ministry data released Thursday.  The grim milestone was passed after 950 new deaths were recorded in the past 24 hours -- the highest single-day increase the country has seen. However, the 10.5% rise is similar to Wednesday’s increase and smaller in percentage terms than any recorded in the past two weeks.  Spain is one of the world's worst-hit countries, trailing only Italy in total deaths from Covid-19, and behind Italy and the United States in total reported cases. Benjamin Netanyahu to self-isolate for a week after minister tests positiveIsrael's Prime Minister Benjamin Netanyahu will enter self-quarantine for seven days, according to a statement from the Prime Minister's Office, after the country's Health Minister tested positive for coronavirus. It is the second time Netanyahu has entered self-quarantine. The 70-year-old Israeli leader self-isolated for a short period after one of his aides tested positive for coronavirus late last month. Netanyahu has twice tested negative for the disease. Health Minister Yaakov Litzman is in good condition, according to a statement from the Health Ministry. The 71-year old and his wife, who also tested positive, will remain at home, where the Health Minister will continue to carry out his job, the statement said. Senior members of the Ministry of Health and aides to Litzman will also self-isolate because of their close contact with the minister. That includes Health Ministry Director-General Moshe Bar Siman Tov, who has frequently held evening briefings with Netanyahu to answer questions and inform the public of new restrictions.  Among others being considered for possible self-isolation by health officials, according to multiple reports on Israeli media, is Yossi Cohen, the head of the Mossad intelligence service. As of Thursday morning, Israel had 6,211 confirmed cases of coronavirus and 32 deaths as a result of the disease. 4 million French workers are on partial unemployment benefits, and the number is rising. From CNN's Barbara Wojazer in Paris. There are now 4 million French employees on partial unemployment support and the number is still “strongly increasing,” France’s Labor Minister Muriel Pénicaud said Thursday. Under the scheme, firms can reduce their activity while asking the state for compensation to be redistributed to employees.Employees on the minimum wage or in a part-time job will receive 100% of their usual salary, the minister said. Other employees will receive 84% of their salary, Pénicaud added. The partial unemployment system should enable the country to recover as quickly as possible after the crisis, as it maintains the link between employers and employees, French PM Edouard Philippe explained on Wednesday."
    phrases=find_phrases(text1)
    voc,vocab=create_vocabulary(text1)
    iv=inverted_index(phrases,voc)
    tf_idf=compute_tf_idf(phrases,voc,iv)
    print_tf_idf(tf_idf,phrases)
    query="catalla, a career diplomat for 27 years, died in hospital in beirut"
    #query="Catalla died in hospital"
    #query="Hqello, i am Salvatore"
    qv=compute_tf_idf_query(len(phrases),query,voc,iv)
    cd=cosine_distance(qv,tf_idf)
    print("Query: %s"%(query))
    print("Distanza Coseno:")
    print_cd(cd)