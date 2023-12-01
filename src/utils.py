from compute_metrics import compute_metrics, exact_match_score
import random, os, nltk
from nltk.corpus import words



def select_random_words(num_rand_words):
    # rand words
    try:
        words_list = words.words()
    except:
        print("NLTK words dataset is downloaded before generating random words.")
        nltk.download('words')
        words_list = words.words()
        
    random_words = ' '.join(random.sample(words_list, num_rand_words))
    
    return random_words  

def extract_random_words(sentence, ratio):
    # trunc-shuf
    # split the sentence into words
    words = sentence.split()[1:]
    
    # compute the number of words to extract
    num_words = int(ratio * len(words))
    
    # select the words considering the computed number
    selected_words = random.sample(words, num_words)
    
    # join the selected words to form a new sentence
    new_sentence = ' '.join(selected_words)
    
    return new_sentence

def compute_em_metrics(predictions, references):
    exact_match = 0
    for pred, ref in zip(predictions, references):
        exact_match += exact_match_score(pred, ref)
    return exact_match


def compute_mmlu_results(total_results):
    '''
    Calculate per subject exact match of total score and subject scores
    '''
    total_em = 0
    total_length = 0
    logging_scores = {}
    for subject, subject_result in total_results.items():
        subject_em = compute_em_metrics(subject_result["prediction"], subject_result["output"])
        logging_scores[subject+"_exact_match"] = 100.0 * subject_em / len(subject_result["output"])
        total_em += subject_em
        total_length += len(subject_result["output"])
    logging_scores["exact_match"] = 100.0 * total_em / total_length
    return logging_scores
    

def compute_superni_results(total_results, task_to_category_dict, category_to_task_dict):
    '''
    Get Rouge-L, Rouge-1, Exact Match Score
    '''
    total_predictions, total_references = [], []
    task_predictions, task_references = {}, {}
    category_predictions, category_references = {}, {}

    for task_name, task_result in total_results.items():
        task_predictions[task_name] = []
        task_references[task_name] = []
        category = task_to_category_dict[task_name]
        
        if category not in category_predictions:
            category_predictions[category] = []
            category_references[category] = []
            
        for _, v in task_result.items():
            total_predictions.append(v["prediction"])
            total_references.append(v["outputs"])
            task_predictions[task_name].append(v["prediction"])
            task_references[task_name].append(v['outputs'])
            category_predictions[category].append(v["prediction"])
            category_references[category].append(v['outputs'])

    logging_scores = compute_metrics(total_predictions, total_references, xlingual=False)
    total_scores = {"total": logging_scores}

    for category in category_to_task_dict.keys():
        category_score = compute_metrics(category_predictions[category], 
                            category_references[category], xlingual=False)
        total_scores[category] = category_score
        logging_scores[category+"_exact_match"] = category_score["exact_match"]
        logging_scores[category+"_rouge1"] = category_score["rouge1"]
        logging_scores[category+"_rougeL"] = category_score["rougeL"]

    for task_name in total_results.keys():
        task_score = compute_metrics(task_predictions[task_name], 
                                    task_references[task_name], xlingual=False)
        total_scores[task_name] = task_score

    return total_results, total_scores, logging_scores

def directory_setter(path="./results", make_dir=False):
    """
    Make dictionary if not exists.
    """
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # make dir if not exist
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )
