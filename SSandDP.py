import nltk
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#requires the models
model=load_model('model_dense_softmax.hdf5')
model_cnn=load_model('model_cnn.hdf5')


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def txtToString(filepath): 
    #reads a text file and returns it as one string
    file=open(filepath,"r",encoding="utf8")
    s=""
    for cnt, line in enumerate(file):
       line=line.rstrip('\n')
       if line=="":
           continue
       if line[0]==' ':
           line=line[1:]
       if line[-1] !=' ':
           line+=" "
       s+=line
    file.close()
    return s[:-1]

def sentences(string): #takes a string representing atext and returns the list of sentences
    #takes a string representing at ext and returns the list of sentences
    return nltk.tokenize.sent_tokenize(string)

def multiSentences(ls,n): #takes a list of sentences and returns all the possible concatenation of following n sentences
    #takes a list of sentences and returns all the possible concatenation of following n sentences
    res=[]
    for i in range(len(ls)-n+1):
        s=""
        for j in range(n-1):
            s+=ls[i+j] + " "
        s+=ls[i+n-1]
        res.append(s)
    return res

def senttoseq(sentences):
    tokenizer=load_obj('tokenizer')
    seq=tokenizer.texts_to_sequences(sentences)
    return pad_sequences(seq, maxlen=300)
    


def loadQuotes(filename):#laod all but three quotes
    return np.genfromtxt(filename,delimiter=";",dtype="str")

def countDifferent(x): #117 sentiments
    l=[]
    count=[]
    for i in range(len(x)):
        if x[i,2] not in l:
            l.append(x[i,2])
            count.append(1)
        else:
            count[l.index(x[i,2])]+=1
    return count,l

def computeLabels():
    count,l=countDifferent(loadQuotes('q2.csv'))
    labels={}
    for i in range(len(l)):
        if count[i]>500 and l[i]!="dad":
            labels[len(labels)]=l[i]
    save_obj(labels,"labels") 

def computeClasses():
    x=loadQuotes()
    l=[]
    for i in range(len(x)):
        if x[i,2] not in l:
            l.append(x[i,2])
    return l



def findNewLabel(label):
    if label in ["dating","love","marriage","romantic","wedding"]:
        return "romance"
    elif label in ["experience","failure","success"]:
        return "achievement"
    elif label in ["diet","fitness","food","health","sports","medical"]:
        return "health"
    elif label in ["education","teacher","graduation","learning"]:
        return "learning"
    elif label in ["hope","future","dreams"]:
        return "future/dreams"
    elif label in ["art","architecture","design"]:
        return "art"
    elif label in ["fear","sad","death"]:
        return "dark toughts"
    elif label in ["government","politics","society","equality"]:
        return "society/politics"
    elif label in ["money","finance"]:
        return "money"
    elif label in ["movies","music"]:
        return "media"
    elif label in ["computers","technology"]:
        return "technology"
    elif label in ["nature","environmental"]:
        return "nature"
    elif label in ["religion","faith","god"]:
        return "religion"
    elif label in ["science","history"]:
        return "science/history"
    elif label in ["truth","wisdom","knowledge"]:
        return "wisdom"
    elif label in ["war","peace","patriotism"]:
        return "war"
    else:
        return ""   


labels={
        "romance" : 2,
        "achievement" : 3,
        "health" : 4,
        "learning" : 5,
        "future/dreams" : 6,
        "art" : 7,
        "dark toughts" : 10,
        "society/politics" : 11,
        "money" : 12,
        "media" : 13,
        "technology" : 14,
        "nature" : 15,
        "religion" : 0,
        "science/history": 8,
        "wisdom" : 1,
        "war" : 9,      
        }


rlabels=dict((v, k) for k, v in labels.items())

def purifyCSV(x):
    newx=[]
    for i in range(len(x)):
        newlabel=findNewLabel(x[i,2])
        if newlabel != "":
            x[i,2]=newlabel
            newx.append(x[i])
    #Write a policy in .policy format
    with open("newquotes6.csv", 'w') as f:
        for i in range(len(newx)):
            f.write(newx[i][0]+";"+newx[i][1]+";"+newx[i][2])
            f.write("\n")
            


def messageToVector(message,dictionary):
    v=np.zeros(len(dictionary))
    for i in range(len(message)):
        if message[i] in dictionary:
            v[dictionary[message[i]]]+=1
    return v

def stemMessage(message):
    message = message.lower()
    message = message.replace(".","")
    message = message.replace(",", "")
    message = word_tokenize(message)
    ps = PorterStemmer()
    stemmed_message = []
    for word in message:
        stemmed_message.append(ps.stem(word))
    return stemmed_message
file = open("dict6.pkl", "rb")
dictionary=pickle.load(file)
file.close()

def createTrainingSet(data,dictionary=dictionary,labels=labels):
    x=np.zeros((len(data),len(dictionary)))
    y=np.zeros((len(data),len(labels)))
    for i in range(len(data)):
        x[i]=messageToVector(stemMessage(data[i,0]),dictionary)
        y[i,labels[data[i,2]]]+=1
    todelete=[]
    for i in range(len(x)):
        if (x[i]==0).all():
            todelete.append(i)
    x=np.delete(x,todelete,axis=0)
    y=np.delete(y,todelete,axis=0)
    o=np.arange(0,len(x))
    np.random.shuffle(o)
    np.save("x_train.npy",x[o])
    np.save("y_train.npy",y[o])

def createsentencesmatrix(sentences,dictionary=dictionary):
    sm=np.zeros((len(sentences),len(dictionary)))
    for i in range(len(sentences)):
        sm[i]=messageToVector(stemMessage(sentences[i]),dictionary)
    return sm

def computePredictions(sentences,model=model,dictionary=dictionary,rlabels=rlabels,addsum=True): 
    #compute the predictions for all the sentencesand the summed score for all classes
    sm=createsentencesmatrix(sentences,dictionary)
    nx,ny=np.shape(sm)
    if addsum:
        predictions=np.zeros((nx,len(rlabels)+1))
        predictions[:,:-1]=model.predict(sm)
        predictions[:,-1]=np.sum(predictions[:,:-1],axis=1)
    else:
        predictions=np.zeros((nx,len(rlabels)))
        predictions=model.predict(sm)
    return predictions

def computePredictionscnn(sentences,model=model_cnn,rlabels=rlabels): 
    #compute the predictions for all the sentencesand the summed score for all classes
    sm=senttoseq(sentences)
    return model.predict(sm)

def findBestQuotePred(sentences,predictions,rlabels=rlabels): #find the "best" quote and it's most important sentiment
    #find the best quote and its major sentiment, where best means highest sum of probabilities
    am=np.argmax(predictions,axis=0)
    
    return sentences[am[-1]],rlabels[np.argmax(predictions[am[-1],0:-1])]

def findQuotesPred(sentences,predictions,rlabels=rlabels,thresold=0.75): #find the best quote for each sentiment that have at least onequote with probe sup at thresold for this sentiment
    #find the best quote for each sentiment such that at least one sentence has a probability of thresold for this sentiment
    d=len(rlabels)
    tab=predictions[:,0:-1]-0.75
    m=np.max(tab,axis=0)
    am=np.argmax(tab,axis=0)
    res=[]
    for i in range(d):
        if m[i]>0:
            res.append([sentences[am[i]],rlabels[i]])
    return res

def findBestQuote(filepath,rlabels=rlabels,maxsentences=1):
    #same as above but from the text file
    stest=txtToString(filepath)
    lsentences=sentences(stest)
    for i in range(2,maxsentences+1):
        lsentences+=multiSentences(lsentences,i)
    predictions=computePredictions(lsentences)
    print(predictions[:,-1])
    return findBestQuotePred(lsentences,predictions)

def findQuotes(filepath,rlabels=rlabels,maxsentences=1,thresold=0.75):
    #same as above but from the text file
    stest=txtToString(filepath)
    lsentences=sentences(stest)
    for i in range(2,maxsentences+1):
        lsentences+=multiSentences(lsentences,i)
    predictions=computePredictions(lsentences)
    
    return findBestQuotePred(lsentences,predictions)

def findThemesoftmax(filepath):
    stest=txtToString(filepath)
    lsentences=sentences(stest)
    predictions=computePredictions(lsentences,addsum=False)
    occlabel=np.zeros(len(labels))
    maxlabel=np.zeros(len(labels))
    indmaxlabel=-np.ones(len(labels))
    for i in range(len(predictions)):
        am=np.argmax(predictions[i])
        occlabel[am]+=1
        if maxlabel[am]<predictions[i,am]:
            maxlabel[am]=predictions[i,am]
            indmaxlabel[am]=i
    return rlabels[np.argmax(occlabel)],lsentences[int(indmaxlabel[np.argmax(occlabel)])]
    

def findThemesoftmax2(filepath,cnn=False):
    stest=txtToString(filepath)
    lsentences=sentences(stest)
    if cnn:
        predictions=computePredictionscnn(lsentences)
    else:
        predictions=computePredictions(lsentences,addsum=False)
    occlabel=np.zeros(len(labels))
    maxlabel=np.zeros(len(labels))
    indmaxlabel=-np.ones(len(labels))
    for i in range(len(predictions)):
        am=np.argmax(predictions[i])
        if predictions[i,am]>0.5:
            occlabel[am]+=1
        if maxlabel[am]<predictions[i,am]:
            maxlabel[am]=predictions[i,am]
            indmaxlabel[am]=i
    print(occlabel)
    return rlabels[np.argmax(occlabel)],lsentences[int(indmaxlabel[np.argmax(occlabel)])]




   
def findThemesigmoid(filepath):
    stest=txtToString(filepath)
    lsentences=sentences(stest)
    predictions=computePredictions(lsentences,addsum=True)
    occlabel=np.zeros(len(labels))
    maxlabel=np.zeros(len(labels))
    indmaxlabel=-np.ones(len(labels))
    for i in range(len(predictions)):
        am=np.argmax(predictions[i,:-1])
        occlabel[am]+=1
        if maxlabel[am]<predictions[i,-1]:
            maxlabel[am]=predictions[i,-1]
            indmaxlabel[am]=i
    return rlabels[np.argmax(occlabel)],lsentences[int(indmaxlabel[np.argmax(occlabel)])]     




