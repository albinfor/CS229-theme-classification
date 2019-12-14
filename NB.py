import csv
import os
import sys
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

class Preprocess:
    def __init__(self,X): #Creates the dictionary of feature words
        self.word_dictionary = self.create_dict(X)
    def create_dict(self,X):
        worddict = {}
        dictionary = {}
        i = 0; percent = 0;
        stopWords = set(stopwords.words('english')) #Implements stopwords
        stopWords = self.get_words(" ".join(stopWords))
        print("Creating dictionary:")
        print("=" * 20)
        for message in X:

            #Prints progress
            i += 1
            if np.rint(100*i/len(X)) != percent:
                percent = np.rint(100*i/len(X))
                if percent%1 == 0:
                    sys.stdout.write(f"\r{int(percent)}%")
                    sys.stdout.flush()
                    if percent == 100: print();
                    #print(f"{percent}%")

            #Counts occurances of words in the message
            message = self.get_words(message)
            previouslist = []
            for word in message:
                if word not in previouslist and word not in stopWords:
                    if word not in worddict.keys():
                        worddict[word] = 0
                    worddict[word] += 1
                    previouslist.append(word)

        #Creates the dictionary if a word exists more than threshold times and returns dictionary.
        threshold = 150
        for key in worddict.keys():
            if worddict[key] > threshold:
                dictionary[key] = len(dictionary)
        print(len(dictionary))
        f=open("dict6.pkl","wb")
        pickle.dump(dictionary,f)
        f.close()
        return dictionary

    def get_words(self,message):
        #Takes a string as input and returns a list consisting of lower case words of stem form
        message = message.lower()
        message = message.replace(".","")
        message = message.replace(",", "")
        message = word_tokenize(message)
        ps = PorterStemmer()
        stemmed_message = []
        for word in message:
            stemmed_message.append(ps.stem(word))
        return stemmed_message


    def transform_text(self,messages):
        #Creates a matrix of features
        featureMatrix = np.zeros((len(messages),len(self.word_dictionary.keys())), dtype=int)

        #Prints progress
        percent = 0
        if len(messages) != 1:
            print("Generating feature vectors from messages:")
            print("=" * 20)
        for counter, message in enumerate( messages):
            if np.rint(100*counter/len(messages)) != percent:
                percent = np.rint(100*counter/len(messages))
                if percent % 1 == 0:
                    sys.stdout.write(f"\r{int(percent)}%")
                    sys.stdout.flush()
                    if percent == 100: print();
            message = self.get_words(message)

            #Converts the words into a feature matrix
            for word in message:
                if word in self.word_dictionary.keys():
                    featureMatrix[counter, self.word_dictionary[word]] += 1
        return featureMatrix

class sentimentpredictor:
    #Naive bayes classifier
    def __init__(self, updateDictionary = True):

        #Variable initialization
        preprocesssavefile = "preprocessfile6.sav"
        self.y = []
        sentences = []
        stop = 0 #Limit number of lines to be read

        #Read from a CSV file
        with open('newquotes6.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                self.y.append(row[2])
                sentences.append(row[0])
                line_count += 1
                if line_count == stop:
                    break
            print(f'{line_count} lines read.')

        #Turn the list into ndarrays
        self.y = np.asarray(self.y)
        sentences = np.asarray(sentences)

        #If updateDictionary then calculate the new dictionary values, otherwise read object from storage
        if updateDictionary or not os.path.isfile(preprocesssavefile):
            file = open(preprocesssavefile, "wb")
            self.preprocesser = Preprocess(sentences)
            pickle.dump(self.preprocesser, file)
            print("Dictionary created")
            print("=" * 20+"\n")
        else:
            file = open(preprocesssavefile, "rb")
            self.preprocesser = pickle.load(file)
            print("=" * 20)
            print("Dictionary Loaded")
            print("=" * 20+"\n")
        file.close()
        #Transform sentences into featurevectors
        self.X = self.preprocesser.transform_text(sentences)
        self.clf = None
        self.labels = None
    def train(self):
        #Train model
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = .1)#, random_state = 0)
        self.clf = MultinomialNB(alpha=10)
        self.clf.fit(X_train, y_train)
        self.labels = self.clf.classes_
        print("Model Trained")
        y_pred = self.clf.predict(X_test)
        y_predtrain=self.clf.predict(X_train)
        
        cf=confusion_matrix(y_test,y_pred)
        np.save("confusion_matrix_NB.npy",cf)
        np.save('x_test_NB.npy',X_test)
        ##Calculating metrics
        print(accuracy_score(y_train,y_predtrain))
        warnings.simplefilter("ignore")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='macro')
        fscore = f1_score(y_test, y_pred, average='macro')
        warnings.simplefilter("default")
        print(f"Model acheives {accuracy} accuracy on test set.")
        print(f"Model acheives {precision} precision on test set.")
        print(f"Model acheives {recall} recall on test set.")
        print(f"Model acheives {fscore} fscore on test set.")
        print("=" * 20)


    def predict(self,sentence):
        #Make predictions on model
        Xpred = self.preprocesser.transform_text([sentence])
        probabilities = self.clf.predict_proba(Xpred.reshape(1, -1))
        return probabilities

    def classes(self):
        l=[]
        for label in self.labels:
            l.append(label)
        return l
        


#True if loading model form memory, otherwise a new model will be generated
usenewmodel = False
#True if you want to updatedictionary before training model
updatedictionary = False

if usenewmodel or not os.path.isfile("sentimentfinder6.sav"):
    #Train model
    sentimentprobabilities2 = sentimentpredictor(updateDictionary = updatedictionary)
    file = open("sentimentfinder6.sav", "wb")
    pickle.dump(sentimentprobabilities2,file)
    print("=" * 20)
    print("New feature matrix used for training")
else:
    file = open("sentimentfinder6.sav", "rb")
    sentimentprobabilities2 = pickle.load(file)
    print("=" * 20)
    print("Previous feature matrix used for training")
file.close()

#Generate prediction
sentimentprobabilities2.train()


#predictions = sentimentprobabilities2.predict("This is an example sentence that we can use to predict different classes, mom mom mom mom")



#Creates a dictionary with probabilities
#a =predictions.flatten().tolist()
#b =sentimentprobabilities.labels.tolist()
#map = zip(b,a)
#print(set(map))