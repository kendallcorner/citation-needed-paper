import re
import argparse
import pandas as pd
import pickle
import numpy as np
import mwapi
from bs4 import BeautifulSoup

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

from keras.utils import to_categorical

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))

'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to determinee whether a statement needs a\
        citation or not.')
    parser.add_argument('-i', '--input', help='The input file from which we\
        read the statements. Lines contains tab-separated values: the\
        statement, the section header, and additionally the binary label\
        corresponding to whether the sentence has a citation or not in the\
        original text. This can be set to 0 if no evaluation is needed.\
        ', required=True)
    parser.add_argument('-o', '--out_dir', help='The output directory where we\
        store the results', required=True)
    parser.add_argument('-m', '--model', help='The path to the model which we\
        use for classifying the statements.', required=True)
    parser.add_argument('-v', '--vocab', help='The path to the vocabulary of\
        words we use to represent the statements.', required=True)
    parser.add_argument('-s', '--sections', help='The path to the vocabulary\
        of section with which we trained our model.', required=True)
    parser.add_argument('-l', '--lang', help='The language that we are parsing\
        now.', required=False, default='en')

    return parser.parse_args()


'''
    Parse and construct the word representation for a sentence.
'''


def text_to_word_list(text):
    # check first if the statements is longer than a single sentence.
    sentences = re.compile('\.\s+').split(str(text))
    if len(sentences) != 1:
        # text = sentences[random.randint(0, len(sentences) - 1)]
        text = sentences[0]

    text = str(text).lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()

    return text


'''
    Compute P/R/F1 from the confusion matrix.
'''


'''
    Create the instances from our datasets
'''


def construct_instance_reasons(statement_path, section_dict_path,
                               vocab_w2v_path, max_len=-1):
    # Load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'), encoding='latin1')

    # load the section dictionary.
    section_dict = pickle.load(open(section_dict_path, 'rb'),
                               encoding='latin1')

    # Load the statements, the first column is the statement and the second is
    # the label (True or False)
    statements = pd.read_csv(statement_path, sep='\t', index_col=None,
                             error_bad_lines=False, warn_bad_lines=False)

    # construct the training data
    X = []
    sections = []
    outstring = []
    for index, row in statements.iterrows():
        try:
            statement_text = text_to_word_list(row['statement'])

            X_inst = []
            for word in statement_text:
                if max_len != -1 and len(X_inst) >= max_len:
                    continue
                if word not in vocab_w2v:
                    X_inst.append(vocab_w2v['UNK'])
                else:
                    X_inst.append(vocab_w2v[word])

            # extract the section, and in case the section does not exist in
            # the model, then assign UNK
            section = row['section'].strip().lower()
            sections.append(
                np.array(
                    [section_dict[section] if section in section_dict else 0]))

            # some of the rows are corrupt, thus, we need to check if the\
            # labels are actually boolean.

            X.append(X_inst)
            outstring.append(str(row["statement"]))
            # entity_id  revision_id timestamp   entity_title    section_id  section prg_idx sentence_idx    statement   citations

        except Exception as e:
            print(row)
            print(e.message)
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

    encoder = LabelBinarizer()

    return X, np.array(sections), encoder, outstring


def splitter(mystring):
    '''Splitter to split types of sentences'''
    mylist = mystring.split(". ")
    mylist = list(map(lambda a: a.split(".[*] "), mylist))
    return [y for x in mylist for y in x]


def get_article_statements(article_name):
    '''get articles using mwapi and Beautiful Soup and format for model'''
    session = mwapi.Session('https://en.wikipedia.org')
    query = session.get(action='query', titles=article_name)
    pages = query['query']['pages']
    soup = None

    if not pages:
        return

    for pageid in pages:
        data = session.get(action='parse', pageid=pageid, prop='text')
        text = data['parse']['text']['*']
        soup = BeautifulSoup(text, 'lxml')
        section = "MAIN_SECTION"
        outstr = 'section\tstatement\n'
        for child in soup.html.body.div.contents:
            if child.name == 'h2':
                section = child.text
            if child.name == 'p':
                statements = splitter(child.text)
                for num, statement in enumerate(statements):
                    if statement == '\n':
                        continue
                    outstr += section + '\t' + statement.replace('\n', '') + '\n'
            if child.name == 'ul':
                for num, list_item in enumerate(child.contents):
                    if list_item == '\n':
                        continue
                    outstr += section + '\t' + list_item.text + '\n'

    fout = open('statements.tsv', 'wt')
    # making a new tsv so I can check the results of the formatting and is
    # quicker to feed into existing script
    fout.write(outstr)
    fout.flush()
    fout.close()


if __name__ == '__main__':
    p = get_arguments()

    # load the model
    model = load_model(p.model)

    # load the data
    max_seq_length = model.input[0].shape[1].value

    get_article_statements(p.input)

    X, sections, encoder, outstring = construct_instance_reasons(
        'statements.tsv', p.sections, p.vocab, max_seq_length)

    # classify the data
    pred = model.predict([X, sections])

    # store the predictions: printing out the sentence text, the prediction
    # score, and original citation label.
    outstr = 'Text\tPrediction\n'

    ordered_list = []
    for idx, y_pred in enumerate(pred):
        list = [outstring[idx], y_pred[0]]
        ordered_list.append(list)

    def sortPred(a):
        return a[1]

    ordered_list.sort(key=sortPred)

    for statement, pred in ordered_list:
        outstr += statement + '\t' + str(pred) + '\n'

    fout = open(p.out_dir + '/' + p.lang + '_predictions_sections.tsv', 'wt')
    fout.write(outstr)
    fout.flush()
    fout.close()
