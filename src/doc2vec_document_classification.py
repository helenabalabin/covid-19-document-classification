# imports
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import os

# create vector representation for the classifier on top of the doc2vec model
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors

# create baseline doc2vec model trained on the abstracts
def d2v_document_classification():
    # code for this function is partially taken from
    # https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4

    # load the dataset, use the same as for the other classifiers
    train_docs = pd.read_csv("../data/document_classification/train.csv", sep="\t")
    test_docs = pd.read_csv("../data/document_classification/test.csv", sep="\t")

    best_f1 = 0
    best_model = ""

    data, labels = train_docs["text"], train_docs["Categories"]
    skf = StratifiedKFold(n_splits=int(1/TESTSIZE))

    # cross validation procedure
    for idx, (train_index, valid_index) in enumerate(skf.split(data, labels)):
        X_train, X_valid = data[train_index], data[valid_index]
        y_train, y_valid = labels[train_index], labels[valid_index]

        # create train splits and test set with annotations
        tagged_tr = [TaggedDocument(doc.split(), [i]) for doc, i in zip(X_train, y_train)]
        tagged_valid = [TaggedDocument(doc.split(), [i]) for doc, i in zip(X_valid, y_valid)]

        # create a doc2vec model
        model = Doc2Vec(vector_size=VECSIZE,
                        window=WINDOWSIZE,
                        min_count=MINCOUNT,
                        workers=8)
        model.build_vocab(tagged_tr)

        # train the model and save it
        model.train(tagged_tr, total_examples=model.corpus_count, epochs=EPOCHS)
        model.save('saved_models/covid-doc2vec-' + str(idx) + '/doc2vec_covid_classification.model')

        # evaluate model on validation set
        y_train, X_train = vector_for_learning(model, tagged_tr)
        y_valid, X_valid = vector_for_learning(model, tagged_valid)

        # create logreg model for final classification
        logreg = LogisticRegression(tol=1e-3, multi_class="multinomial", max_iter=200)
        logreg.fit(X_train, y_train)
        preds = logreg.predict(X_valid)

        # save the logreg model
        pickle.dump(logreg, open('saved_models/covid-doc2vec-' + str(idx) + '/logreg.model', 'wb'))

        # check if it's the best model
        f1 = f1_score(y_valid, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_model = 'covid-doc2vec-' + str(idx)

        os.makedirs(os.path.dirname('saved_models/covid-doc2vec/classification_report.txt'), exist_ok=True)
        with open('saved_models/covid-doc2vec/classification_report.txt', "a+") as file:
            file.write("Evaluation on withheld split for numfold no. {} \n".format(idx))
            file.write(classification_report(y_valid, preds))
            file.write("\n\n")
            file.close()

    # reload best model for final evaluation on test set
    model = Doc2Vec.load('saved_models/' + best_model + '/doc2vec_covid_classification.model')
    logreg = pickle.load(open('saved_models/' + best_model + '/logreg.model', 'rb'))
    tagged_test = [TaggedDocument(doc.split(), [i]) for doc,i in zip(test_docs["text"], test_docs["Categories"])]
    y_test, X_test = vector_for_learning(model, tagged_test)

    # final predictions
    preds = logreg.predict(X_test)

    with open('saved_models/covid-doc2vec/classification_report.txt', "a+") as file:
        file.write("Final result of the best model \n")
        file.write(classification_report(y_test, preds))
        file.write("\n\n")
        file.close()

    # save the best models again
    model.save('saved_models/covid-doc2vec/doc2vec_covid_classification.model')
    pickle.dump(model, open('saved_models/covid-doc2vec/logreg.model', 'wb'))

    return model

if __name__ == "__main__":
    # defining some properties of the doc2vec model
    VECSIZE = 300
    WINDOWSIZE = 5
    MINCOUNT = 2
    EPOCHS = 20
    # and some other attributes
    TESTSIZE = 0.2

    d2v_document_classification()