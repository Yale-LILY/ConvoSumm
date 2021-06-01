import os
import random

from recap_am.adu import run_task
from recap_am.controller.preprocess import read_files, prep_production
from recap_am.model.config import Config
import recap_am.adu.utilities as utils

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
args = parser.parse_args()

mode = args.mode


def single_run(input_files, label_files, test_files, test_labels):
    doc = read_files(input_files, label_files)
    print("Read Files")
    tr_result = run_task.run_train(doc)
    print("Trained")
    t_doc = read_files(test_files, test_labels)
    _, _, _, _ = run_task.run_test(t_doc, tr_result)
    print("Tested")


def LOOCV(input_files, label_files, test_files, test_labels, n_runs=5):
    avg_adu_acc = 0.0
    avg_adu_prec = 0.0
    avg_adu_rec = 0.0
    avg_adu_f1 = 0.0
    for i in range(n_runs):
        print(i)
        test_files = test_files[-1]
        test_labels = test_labels[-1]
        doc = read_files(input_files, label_files)
        print("Read Files")
        tr_result = run_task.run_train(doc)
        print("Trained")
        t_doc = read_files(test_files, test_labels)
        acc, prec, rec, f1 = run_task.run_test(t_doc, tr_result)
        print("Tested")
        avg_adu_acc += acc
        avg_adu_prec += prec
        avg_adu_rec += rec
        avg_adu_f1 += f1
    avg_adu_acc *= 1 / n_runs
    avg_adu_prec *= 1 / n_runs
    avg_adu_rec *= 1 / n_runs
    avg_adu_f1 *= 1 / n_runs
    print("Avg Accuracy:\tADU\t%8.8f" % (avg_adu_acc))
    print("Avg Precision:\tADU\t%8.8f" % (avg_adu_prec))
    print("Avg Recall:\tADU\t%8.8f" % (avg_adu_rec))
    print("Avg F1-Score:\tADU\t%8.8f" % (avg_adu_f1))


def read(in_files, l_files, input_file):
    if os.path.isdir(input_file):
        for file in os.listdir(input_file):
            in_files, l_files = read(in_files, l_files, input_file + "/" + file)
    else:
        if input_file.endswith(".text"):
            in_files.append(input_file)
            l_files.append(input_file.replace(".text", ".label"))
    return in_files, l_files


config = Config.get_instance()
in_files = []
l_files = []
t_files = []
t_labels = []
datapath = config["adu"]["path"]["input"]
train_path = datapath + "/train"
test_path = datapath + "/test"

for in_file in os.listdir(train_path):
    in_files, l_files = read(in_files, l_files, train_path + "/" + in_file)

for in_file in os.listdir(test_path):
    t_files, t_labels = read(t_files, t_labels, test_path + "/" + in_file)

if mode == "LOOCV":
    LOOCV(in_files, l_files, t_files, t_labels, n_runs=2)
elif mode == "single":
    single_run(in_files, l_files, t_files, t_labels)

