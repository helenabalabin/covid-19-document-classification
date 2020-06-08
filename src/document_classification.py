# imports
from pathlib import Path
import pandas as pd
import json
import numpy as np
from json import JSONEncoder
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import MultiLabelTextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer, EarlyStopping
from farm.utils import set_all_seeds, initialize_device_settings, MLFlowLogger
from sklearn.model_selection import StratifiedKFold
from farm.eval import Evaluator
from sklearn.metrics import f1_score
from datetime import datetime
import os


# helper class for proper JSON formatting
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# cross-validation document classification
def doc_classification_crossvalidation():
    # TODO: reference to FARM example script

    # for local logging:
    ml_logger = MLFlowLogger(tracking_uri="")
    ml_logger.init_experiment(experiment_name="covid-document-classification",
                              run_name=RUNNAME)

    # model settings
    xval_folds = FOLDS
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    if RUNLOCAL:
        device = "cpu"
    n_epochs = NEPOCHS
    batch_size = BATCHSIZE
    evaluate_every = EVALEVERY
    lang_model = MODELTYPE
    do_lower_case = False

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    metric = "f1_macro"

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # The processor wants to know the possible labels ...
    label_list = LABELS
    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=MAXLEN,
                                            data_dir=DATADIR,
                                            train_filename=TRAIN,
                                            test_filename=TEST,
                                            dev_split=0.1,
                                            label_list=label_list,
                                            metric=metric,
                                            label_column_name="Categories",
                                            # confusing parameter name: it should be called multiCLASS
                                            # not multiLABEL
                                            multilabel=True
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # Load one silo for each fold in our cross-validation
    silos = DataSiloForCrossVal.make(data_silo, n_splits=xval_folds)

    # the following steps should be run for each of the folds of the cross validation, so we put them
    # into a function
    def train_on_split(silo_to_use, n_fold, save_dir, dev):
        # Create an AdaptiveModel
        # a) which consists of a pretrained language model as a basis
        language_model = LanguageModel.load(lang_model)
        # b) and a prediction head on top that is suited for our task => Text classification
        prediction_head = MultiLabelTextClassificationHead(
            # there is still an error with class weights ...
            # class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
            num_labels=len(label_list))

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.2,
            lm_output_types=["per_sequence"],
            device=dev)

        # Create an optimizer
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=0.5e-5,
            device=dev,
            n_batches=len(silo_to_use.loaders["train"]),
            n_epochs=n_epochs)

        # Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
        # Also create an EarlyStopping instance and pass it on to the trainer
        save_dir = Path(str(save_dir) + f"-{n_fold}")
        # unfortunately, early stopping is still not working
        earlystopping = EarlyStopping(
            metric="f1_macro", mode="max",
            save_dir=save_dir,  # where to save the best model
            patience=5 # number of evaluations to wait for improvement before terminating the training
        )

        trainer = Trainer(model=model, optimizer=optimizer,
                          data_silo=silo_to_use, epochs=n_epochs,
                          n_gpu=n_gpu, lr_schedule=lr_schedule,
                          evaluate_every=evaluate_every,
                          device=dev, evaluator_test=False,
                          #early_stopping=earlystopping)
                          )
        # train it
        trainer.train()
        trainer.model.save(save_dir)
        return trainer.model

    # for each fold, run the whole training, earlystopping to get a model, then evaluate the model
    # on the test set of each fold
    # Remember all the results for overall metrics over all predictions of all folds and for averaging
    allresults = []
    all_preds = []
    all_labels = []
    bestfold = None
    bestf1_macro = -1
    save_dir = Path("saved_models/covid-classification-v1")

    for num_fold, silo in enumerate(silos):
        model = train_on_split(silo, num_fold, save_dir, device)

        # do eval on test set here (and not in Trainer),
        #  so that we can easily store the actual preds and labels for a "global" eval across all folds.
        evaluator_test = Evaluator(
            data_loader=silo.get_data_loader("test"),
            tasks=silo.processor.tasks,
            device=device,
        )
        result = evaluator_test.eval(model, return_preds_and_labels=True)

        os.makedirs(os.path.dirname(BESTMODEL + "/classification_report.txt"), exist_ok=True)
        with open(BESTMODEL + "/classification_report.txt", "a+") as file:
            file.write("Evaluation on withheld split for numfold no. {} \n".format(num_fold))
            file.write(result[0]["report"])
            file.write("\n\n")
            file.close()

        evaluator_test.log_results(result, "Test", steps=len(silo.get_data_loader("test")), num_fold=num_fold)

        allresults.append(result)
        all_preds.extend(result[0].get("preds"))
        all_labels.extend(result[0].get("labels"))

        # keep track of best fold
        f1_macro = result[0]["f1_macro"]
        if f1_macro > bestf1_macro:
            bestf1_macro = f1_macro
            bestfold = num_fold

    # Save the per-fold results to json for a separate, more detailed analysis
    with open("../data/predictions/covid-classification-xval.results.json", "wt") as fp:
        json.dump(allresults, fp, cls=NumpyArrayEncoder)

    # calculate overall f1 score across all folds
    xval_f1_macro = f1_score(all_labels, all_preds, average="macro")
    ml_logger.log_metrics({"f1 macro across all folds": xval_f1_macro}, step=None)

    # test performance
    evaluator_origtest = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )
    # restore model from the best fold
    lm_name = model.language_model.name
    save_dir = Path(f"saved_models/covid-classification-v1-{bestfold}")
    model = AdaptiveModel.load(save_dir, device, lm_name=lm_name)
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    result = evaluator_origtest.eval(model)
    ml_logger.log_metrics({"f1 macro on final test set": result[0]["f1_macro"]}, step=None)

    with open(BESTMODEL + "/classification_report.txt", "a+") as file:
        file.write("Final result of the best model \n")
        file.write(result[0]["report"])
        file.write("\n\n")
        file.close()

    ml_logger.log_artifacts(BESTMODEL + "/")

    # save model for later use
    processor.save(BESTMODEL)
    model.save(BESTMODEL)
    return model


# use the metadata to generate the document classification inputs
# and make train/test splits
def create_textclassification_csv(save=True):
    metafile = pd.read_csv(METADIR, index_col=None, usecols=["PMID", "Abstract", "Categories"])

    # rename Abstract column to fit the FARM input format
    metafile.rename(columns={"Abstract": "text"}, inplace=True)

    # only use the entries with the labels that are included in LABELS
    metafile = metafile[metafile["Categories"].isin(LABELS)]

    skf = StratifiedKFold(n_splits=int(1/TESTSPLIT))
    idx = list(skf.split(X=metafile[["PMID", "text"]],
                            y=metafile["Categories"]))[0]
    train = metafile.iloc[idx[0]]
    test = metafile.iloc[idx[1]]

    # smaller datasets for local execution (useful for checking possible errors in the code
    # before running the code on the cluster)
    mini_train = train[:40]
    mini_test = test[:10]

    if save:
        train.to_csv("../data/document_classification/train.csv", header=["PMID", "text", "Categories"], index=None, sep="\t")
        test.to_csv("../data/document_classification/test.csv", header=["PMID", "text", "Categories"], index=None, sep="\t")

        mini_train.to_csv("../data/document_classification/minitestcase/train.csv", header=["PMID", "text", "Categories"], index=None, sep="\t")
        mini_test.to_csv("../data/document_classification/minitestcase/test.csv", header=["PMID", "text", "Categories"], index=None, sep="\t")

    return train, test, mini_train, mini_test


if __name__ == "__main__":
    # TODO: data exploration ipynb with some basic stats: class distribution, distribution of abstract lengths
    # TODO: documentation

    # TODO: use & compare multiple models
    # testing out different MODELTYPEs
    # 1. bert-base-uncased
    # 2. monologg/biobert_v1.1_pubmed
    # 3. deepset/covid_bert_base
    MODELTYPE = "bert-base-uncased"
    TESTSPLIT = 0.2
    MINLEN = 100
    METADIR = "../data/litcovid_meta/litcovid_mappings.csv"

    # take first split if no best split is specified later or use this variable to specify the
    # model used for inference
    # BESTMODEL = ...
    RUNNAME = "run-" + str(datetime.now().strftime("%m%d%Y-%H:%M:%S"))
    BESTMODEL = "saved_models/covid-classification-best-model-" \
                + str(datetime.now().strftime("%m%d%Y-%H:%M:%S"))
    OUTPUTPATH = "../data/predictions/"
    TRAIN = "train.csv"
    TEST = "test.csv"

    # leave out "General Info" (1% of labels)
    # and "Epidemic Forecasting" (1.9% of labels)
    # and "Transmission" (3.4% of labels)
    LABELS = ["Mechanism", "Treatment", "Case Report", "Diagnosis",  "Prevention"]
    # change this argument to True if you want to execute the code locally (with smaller hyperparameter settings)
    RUNLOCAL = True

    create_textclassification_csv()

    if RUNLOCAL:
        DATADIR = "../data/document_classification/minitestcase/"
        NEPOCHS = 1
        MAXLEN = 32
        BATCHSIZE = 4
        FOLDS = 2
        EVAL = 4
        EVALEVERY = 50

    else:
        DATADIR = "../data/document_classification/"
        # length of words in abstract go roughly from 100 to 300
        MAXLEN = 300
        NEPOCHS = 60
        BATCHSIZE = 64
        FOLDS = 5
        EVAL = 100
        EVALEVERY = 500

    doc_classification_crossvalidation()



