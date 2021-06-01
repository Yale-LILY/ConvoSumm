import os
import random
import copy
import numpy as np
from recap_am.adu import run_task, classify, utilities
from recap_am.controller.preprocess import read_files, prep_production
from recap_am.model.config import Config
import recap_am.adu.utilities as utils
from recap_am.adu.feature_select import filter_feats, add_embeddings

# from recap_am.relation import attack_support, adu_position
import recap_am.relation.controller.pairwise_comparison as pc
from sklearn.utils import shuffle
import argparse

# from recap_am.app import _export

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--mode",
    default="single",
    type=str,
    help="Single or LOOCV run",
    choices=["LOOCV", "single"],
)
parser.add_argument(
    "-a",
    "--adu",
    default="true",
    type=str,
    help="Classified or True ADUs",
    choices=["classified", "true"],
)
args = parser.parse_args()

mode = args.mode
adu = args.adu
# from recap_am.app import _export


def single_run(input_files, label_files, test_files, test_labels, adu_mode="true"):

    doc = read_files(input_files, label_files)
    print("Read Train Files")
    cl_result = run_task.run_clpr_train(doc)
    print("Finished Training")

    t_doc = read_files(test_files, test_labels)
    print("Read Test Files")
    if adu_mode == "true":
        acc, prec, rec, f1 = run_task.run_clpr_test(t_doc, cl_result)
    elif adu_mode == "classified":
        orig_adus_labels = t_doc._.Labels
        t_doc = filter_feats(t_doc, load=True)
        print("Filtered feats")
        t_doc = add_embeddings(t_doc)
        print("Added Embdes")
        t_doc = classify.predict(t_doc)
        clpr_feats = []
        for idx, l in enumerate(t_doc._.Labels):
            if l == 1:
                clpr_feats.append(t_doc._.Features[idx])
        t_doc._.CLPR_Features = clpr_feats
        feature = t_doc._.CLPR_Features
        label = t_doc._.CLPR_Labels
        feature = np.asarray(feature)
        predictions = cl_result.predict(feature).tolist()
        cl_iter = 0
        correct_count = 0
        for idx, l in enumerate(orig_adus_labels):
            if cl_iter < len(predictions):
                if orig_adus_labels[idx] == 1:
                    if orig_adus_labels[idx] == t_doc._.Labels[idx]:
                        if label[cl_iter] == predictions[cl_iter]:
                            correct_count += 1
                    cl_iter += 1
            else:
                break
        prec = correct_count / len(orig_adus_labels)
        print("Precision:\tCLPR\t%8.8f" % (prec))


def LOOCV(input_files, label_files, test_files, test_labels, n_runs=5, adu_mode="true"):
    avg_cl_acc = 0.0
    avg_cl_prec = 0.0
    avg_cl_rec = 0.0
    avg_cl_f1 = 0.0
    for i in range(n_runs):
        print(i)
        input_files, label_files = shuffle(input_files, label_files)
        test_files = test_files[-1]
        test_labels = test_labels[-1]

        doc = read_files(input_files, label_files)
        print("Read Train Files")
        cl_result = run_task.run_clpr_train(doc)
        print("Finished Training")

        t_doc = read_files(test_files, test_labels)
        print("Read Test Files")
        if adu_mode == "true":
            acc, prec, rec, f1 = run_task.run_clpr_test(t_doc, cl_result)
            avg_cl_acc += acc
            avg_cl_prec += prec
            avg_cl_rec += rec
            avg_cl_f1 += f1
        elif adu_mode == "classified":
            orig_adus_labels = t_doc._.Labels
            orig_adus = t_doc._.ADU_Sents
            t_doc = filter_feats(t_doc, load=True)
            print("Filtered feats")
            t_doc = add_embeddings(t_doc)
            print("Added Embdes")
            t_doc = classify.predict(t_doc)
            clpr_feats = []
            for idx, l in enumerate(t_doc._.Labels):
                if l == 1:
                    clpr_feats.append(t_doc._.Features[idx])
            t_doc._.CLPR_Features = clpr_feats
            feature = t_doc._.CLPR_Features
            label = t_doc._.CLPR_Labels
            feature = np.asarray(feature)
            predictions = cl_result.predict(feature).tolist()
            cl_iter = 0
            correct_count = 0
            for idx, l in enumerate(orig_adus_labels):
                if orig_adus_labels[idx] == 1:
                    if orig_adus_labels[idx] == t_doc._.Labels[idx]:
                        if label[cl_iter] == predictions[cl_iter]:
                            correct_count += 1
                    cl_iter += 1
            acc = correct_count / len(orig_adus)
            avg_cl_acc += prec
    avg_cl_acc *= 1 / n_runs
    print("Avg Accuracy:\tCLPR\t%8.8f" % (avg_cl_acc))
    if adu_mode == "true":
        avg_cl_prec *= 1 / n_runs
        avg_cl_rec *= 1 / n_runs
        avg_cl_f1 *= 1 / n_runs
        print("Avg Precision:\tCLPR\t%8.8f" % (avg_cl_prec))
        print("Avg Recall:\tCLPR\t%8.8f" % (avg_cl_rec))
        print("Avg F1-Score:\tCLPR\t%8.8f" % (avg_cl_f1))


def read(in_files, l_files, input_file):
    if os.path.isdir(input_file):
        for file in os.listdir(input_file):
            in_files, l_files = read(in_files, l_files, input_file + "/" + file)
    else:
        if input_file.endswith(".text"):
            in_files.append(input_file)
            l_files.append(input_file.replace(".text", ".label"))
    return in_files, l_files


in_files = []
l_files = []
t_files = []
t_labels = []
config = Config.get_instance()
datapath = "data/ADU/in/PE/" + config["nlp"]["language"]
train_path = datapath + "/train"
test_path = datapath + "/test"
for in_file in os.listdir(train_path):
    in_files, l_files = read(in_files, l_files, train_path + "/" + in_file)

for in_file in os.listdir(test_path):
    t_files, t_labels = read(t_files, t_labels, test_path + "/" + in_file)

if mode == "LOOCV":
    LOOCV(in_files, l_files, t_files, t_labels, adu_mode=adu)
elif mode == "single":
    single_run(in_files, l_files, t_files, t_labels, adu_mode=adu)
