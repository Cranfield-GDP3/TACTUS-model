import json
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot import confusion_matrix as plt_CM
import matplotlib.pyplot as plt
from tqdm import tqdm

AVAILABLE_CLASSES = ['kicking', 'punching', 'pushing', 'neutral']


def get_model_stat(model_data_path: Path):
    return json.load(model_data_path.open(mode='r'))


def get_accuracy(pred, truth):
    accuracy = 0
    for i in range(len(truth)):
        if pred[i] == truth[i]:
            accuracy += 1
    return accuracy / len(truth)


def get_fp_tp(truth, pred, label):
    fp = 0
    tp = 0
    fn = 0
    for i in range(0, len(truth) - 1):
        if pred[i] == label and truth[i] == label:
            tp += 1
        elif pred[i] != label and truth[i] == label:
            fn += 1
        elif pred[i] == label and truth[i] != label:
            fp += 1
        else:
            a = 1
    return tp, fp, fn


def precision(TP, FP):
    if TP == 0:
        return 0
    else:
        return TP / (TP + FP)


def recall(TP, FN):
    if TP == 0:
        return 0
    else:
        return TP / (TP + FN)


def F1_score(P, R):
    if P == 0 and R == 0:
        return 0
    else:
        return 2 * P * R / (P + R)


def model_visualisation(model_stat: dict):
    """
    Plot the confusion matrix and data about the model
    """
    # confusion matrix
    C_M_train = confusion_matrix(model_stat["y_true_train"],
                                 model_stat["y_pred_train"])
    C_M_test = confusion_matrix(model_stat["y_true_test"],
                                model_stat["y_pred_test"])
    CM_visualizer = plt_CM.ConfusionMatrixDisplay(C_M_train, display_labels=AVAILABLE_CLASSES)
    CM_visualizer.plot()
    plt.title("Training Confusion Matrix")
    plt.show()
    CM_visualizer = plt_CM.ConfusionMatrixDisplay(C_M_test, display_labels=AVAILABLE_CLASSES)
    CM_visualizer.plot()
    plt.title("Testing Confusion Matrix")
    plt.show()
    accuracy = get_accuracy(model_stat["y_true_test"],
                            model_stat["y_pred_test"])
    # data
    print("Accuracy : ", accuracy)
    print(model_stat["classifier_name"])
    print(model_stat["hyperparams"])
    print(model_stat["tracker_params"])
    print(model_stat["augment_grid"])


def evaluate(model_stat: dict, print_flag: bool = False):
    """
    Evaluate one model and print data if flag turned on
    """
    train_acc = get_accuracy(model_stat["y_true_train"], model_stat["y_pred_train"])
    test_acc = get_accuracy(model_stat["y_true_test"], model_stat["y_pred_test"])
    list_f1_score = []
    list_recall = []
    list_precision = []
    for label in range(len(AVAILABLE_CLASSES)):
        tp, fp, fn = get_fp_tp(model_stat["y_true_test"], model_stat["y_pred_test"], label)
        recallval = recall(tp, fn)
        precisionval = precision(tp, fp)
        f1_scoreval = F1_score(precisionval, recallval)
        list_f1_score.append(f1_scoreval)
        list_recall.append(recallval)
        list_precision.append(precisionval)
        if print_flag:
            print(AVAILABLE_CLASSES[label], " F1 score: ", round(f1_scoreval, 3))
    avg_scoreval = sum(list_f1_score) / len(list_f1_score)
    avg_recall = sum(list_recall) / len(list_recall)
    avg_precision = sum(list_precision) / len(list_precision)
    if print_flag:
        print("Average F1 score:", round(avg_scoreval, 3))
        print("Average Precision:", round(avg_precision, 3))
        print("Average Recall:", round(avg_recall, 3))
        print("Training accuracy: ", round(train_acc, 3))
        print("Testing accuracy: ", round(test_acc, 3))
    return avg_scoreval

def whoisbest(model_data_folder_path: Path = Path("data\models\evaluation")):
    """

    Args:
        model_data_folder_path: Path of model .json files

    Returns:
        plot and print performances and hyperparameters of the best model optimizing Macro F1 Score
    """
    best_model = None
    path_best = None

    for i in tqdm(model_data_folder_path.iterdir()):
        model_stat = get_model_stat(i)
        if best_model == None:
            best_model = model_stat
            path_best = i

        if evaluate(model_stat) > evaluate((best_model)):
            path_best = i
            best_model = model_stat
    evaluate(best_model, True)
    print("Best model path : ", path_best)

    model_visualisation(best_model)
