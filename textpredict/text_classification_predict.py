from common import file
from model.SVM_Model import SVM_Model
from model.NaiveBayes_Model import NaiveBayes_Model
from model.KNeighborsClassifier_Model import KNeighborsClassifier_Model
from model.DecisionTreeClassifier_Model import DecisionTreeClassifier_Model
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV

import torch

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextClassificationPredict(object):
    def __init__(self, question_test, db_train, db_train_extend, db_answers):

        #declare the data sets
        self.question_test = question_test # the input
        self.db_train = db_train   # Intent set
        self.db_answers = db_answers # Answers set
        self.db_train_extend = db_train_extend # The set of questions extended from the set of intents
        
        # ensemble the models
        self.models = {
            'SVM': SVM_Model().clf,
            'DT' : DecisionTreeClassifier_Model().clf,
            'NaiveBayes': NaiveBayes_Model().clf,
            'KNN': KNeighborsClassifier_Model().clf
        }

        # Initialize VotingClassifier with the individual models
        self.voting_clf = VotingClassifier(
            estimators=[
                ('svm', self.models['SVM']),
                ('dt', self.models['DT']),
                ('nb', self.models['NaiveBayes']),
                ('knn', self.models['KNN'])
            ],
            voting='soft'  # Use 'soft' voting to average the predicted probabilities
        )
    # Text_Predict: Return the response predicted by the model
    def Text_Predict(self):
        fallback_intent = file.get_fallback_intent() # fallback set

        df_train_extend = pd.DataFrame(self.db_train_extend)
        df_train = pd.DataFrame(self.db_train)
        df_answers = pd.DataFrame(self.db_answers) 

        db_Predict = []
        db_Predict.append({"Question": self.question_test})
        df_Predict = pd.DataFrame(db_Predict)


        # train model
        self.voting_clf.fit(df_train_extend["Question"], df_train_extend.Intent)
        
        # Predict probabilities for the test question
        list_score = self.voting_clf.predict_proba(df_Predict["Question"]).flatten()
        
        # print("---list_score:", list_score)
        predicted = list_score.tolist().index(list_score.max())
        # print("---predicted:", predicted)

        if list_score[predicted] >= 0.4:
            mess = df_answers["Answers"][predicted]
            print("Chatbot: " + mess)
            print(list_score[predicted])
        else:
            # If the meaning of the question is unclear, the function will automatically return the mess available in the fallback_intent list
            mess = fallback_intent[random.randint(0, len(fallback_intent) - 1)]
            print(list_score[predicted])
            print(mess)

        return mess
    
    def GridSearchCV(self):
        df_train_extend = pd.DataFrame(self.db_train_extend)
        X = df_train_extend.Question
        y = df_train_extend.Intent

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=0)

        # Define the parameters for the individual models
        parameters = {
            'svm__clf__C': [1e3, 1e4],
            'svm__clf__gamma': [0.0001, 0.001],
            'dt__clf__max_depth': range(1, 10),
            'knn__clf__n_neighbors': range(1, 10),
            'nb__clf__alpha': (1, 0.1, 0.01)
        }

        # Using GridSearchCV + Voting Classifier
        cv = RandomizedSearchCV(self.voting_clf, parameters, n_iter=10, cv=3, verbose=2, n_jobs=-1)
        t0 = time()
        cv.fit(X_train, y_train)

        print("Done in {0}s".format(time() - t0))

        # Show the best parameters
        print('Best parameters found: ', cv.best_params_)

        # Evaluate accuracy on training and testing sets
        y_train_pred = classification_report(y_train, cv.predict(X_train))
        y_test_pred = classification_report(y_test, cv.predict(X_test))

        print("""{model_name}\n Train Accuracy: \n{train} 
                \n Test Accuracy:  \n{test}""".format(model_name="Voting Classifier",
                                                    train=y_train_pred, test=y_test_pred))
 
    def Test_Model_ByNormal(self):
        df_train_extend = pd.DataFrame(self.db_train_extend)
        X = df_train_extend.Question
        y = df_train_extend.Intent

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=1)

        self.voting_clf.fit(X_train, y_train)  # Training Model

        # VotingClassifier Training Model 
        y_train_pred = classification_report(y_train, self.voting_clf.predict(X_train), digits=4)
        y_test_pred = classification_report(y_test, self.voting_clf.predict(X_test), digits=4)

        print("""\n Train Accuracy: \n{train} 
        \n Test Accuracy:  \n{test}""".format(
                                              train=y_train_pred, test=y_test_pred))

    def Test_Model_ByLeaveOneOut(self):
        df_train_extend = pd.DataFrame(self.db_train_extend)
        X = df_train_extend.Question.values  # change to numpy array
        y = df_train_extend.Intent.values

        clf = self.voting_clf
        cv = LeaveOneOut()

        _precision_score = 0
        _recall_score = 0
        _accuracy = 0
        count = 0

        for train_index, test_index in cv.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            clf.fit(X_train.reshape(-1, 1), y_train)

            pred_train = clf.predict(X_train.reshape(-1, 1))
            _precision_score += precision_score(y_train, pred_train, average='weighted')
            _recall_score += recall_score(y_train, pred_train, average='weighted')

            pred_test = clf.predict(X_test.reshape(-1, 1))
            _accuracy += accuracy_score(y_test, pred_test)

            count += 1
            print(f"Iteration {count}")

            # Classification report for each iteration (optional)
            print("Train Accuracy:\n", classification_report(y_train, pred_train))
            print("Test Accuracy:\n", classification_report(y_test, pred_test))

        # Calculate macro-averages
        MacroAVG_Precision_Score = _precision_score / count
        MacroAVG_Recall_Score = _recall_score / count
        MacroAVG_FScore = 2 * MacroAVG_Precision_Score * MacroAVG_Recall_Score / (MacroAVG_Precision_Score + MacroAVG_Recall_Score)

        print("Macro-average Precision-Score: ", MacroAVG_Precision_Score)
        print("Macro-average Recall-Score: ", MacroAVG_Recall_Score)
        print("Macro-average F-Score: ", MacroAVG_FScore)
        print("Accuracy: ", _accuracy / count)
    
if __name__ == '__main__':
    # Processing input
    print("Xin chào, tôi có thể giúp gì cho bạn ? (nhập 'stop' để dừng)")
    while True:
        sentence = input("Bạn: ")
        if sentence == "stop":
            break

        predict = TextClassificationPredict(sentence, file.get_dbtrain(), file.get_dbtrain_extend(), file.get_dbanswers()) # Class initialization  
        # predict.GridSearchCV()
        # predict.Test_Model_ByNormal()
        # predict.Test_Model_ByLeaveOneOut()
        predict.Text_Predict() # make predictions