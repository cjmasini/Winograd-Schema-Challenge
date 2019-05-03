import spacy
import numpy as np
from conceptNet import conceptNet
from load_questions import load_data

listy = []

'''
Use spacy model to tag the key word of each answer, using adjectives or proper
nouns with the root word if each answer has the same getroot.
'''
def parse_candidate(candidateA, candidateB, pdps_nlp):
    doc_A = pdps_nlp(candidateA)
    doc_B = pdps_nlp(candidateB)

    if doc_A.__len__ == 1:
        pre_candidateA = doc_A[0]
    else:
        for token in doc_A:
            if token.dep_ == 'ROOT':
                pre_candidateA = token
                break

    if doc_B.__len__ == 1:
        pre_candidateB = doc_B[0]
    else:
        for token in doc_B:
            if token.dep_ == 'ROOT':
                pre_candidateB = token
                break

    if pre_candidateA.text == pre_candidateB.text:
        for token in doc_A:
            if token.head.dep_ =='ROOT' and (token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
                pre_candidateA = token
        for token in doc_B:
            if token.head.dep_ =='ROOT' and (token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
                pre_candidateB = token

    return pre_candidateA, pre_candidateB

'''
Use spacy model to tag the key word of each pronoun to be resolved
'''
def parse_pronoun(pronoun, pdps_nlp):
    doc_pronoun = pdps_nlp(pronoun)
    if doc_pronoun.__len__ == 1:
        return str(doc_pronoun[0])
    else:
        for token in doc_pronoun:
            if token.dep_ == 'ROOT':
                return str(token)
                break

'''
After drive the key word of each candidate, we need embed those key
word into problem statement, so we could access dependency parse,
pos in the doc_statement as well as the doc_tensor
'''
def embed_candidate(pre_candidateA, pre_candidateB, pronoun,doc_state):
    for token in doc_state:
        if token.text.lower() == pronoun.lower():
            token_pronoun = token

    #find candidate A in context
    for token in doc_state:
        if pre_candidateA.lemma == token.lemma:
            token_candidateA = token
            break
    if token.i == doc_state.__len__() - 1:
        similarity = 0
        for token_2 in doc_state:
            if token_2.similarity(pre_candidateA) > similarity:
                similarity = token_2.similarity(pre_candidateB)
                token_candidateA = token_2

    #find candidate B in context
    for token in doc_state:
        if pre_candidateB.lemma == token.lemma:
            token_candidateB = token
            break
    if token.i == doc_state.__len__() - 1:
        similarity = -100
        for token_2 in doc_state:
            if token_2.similarity(pre_candidateB) > similarity:
                similarity = token_2.similarity(pre_candidateB)
                token_candidateB = token_2

    return token_candidateA, token_candidateB, token_pronoun

'''
Use tensors to do similarity estimate

Because spaCy uses a 4-layer convolutional network to processing doc,
spacy will ncodes a document's internal meaning representations as an
array of floats, also called a tensor. The tensors are sensitive to
up to four words on either side of a word.
'''

def tensor_similarity(token_candidate, token_pronoun, doc):
    vector_candidate = doc.tensor[token_candidate.i]
    vector_pron = doc.tensor[token_pronoun.i]

    norm_candidate = np.linalg.norm(vector_candidate)
    norm_pron =np.linalg.norm(vector_pron)

    similarity = np.dot(vector_candidate, vector_pron)/(norm_candidate * norm_pron)
    return similarity


'''
How can we use commonsense knowledge to parse the lauguage
# def word_embedding_with_knowledge():
# https://github.com/iunderstand/SWE
learn from the conceptNET and try to understand the relation rule-base
'''
def analysis_commensense(token_candidateA, token_candidateB, token_pronoun, doc_state):
    answer = None
    key1_token = token_candidateA
    key2_token = token_candidateB

    for token in doc_state[:token_pronoun.i]:
        if token.dep_ == 'ROOT' or token.dep_ == 'xcomp' or token.dep_ == 'ccomp' \
            or token.dep_ == 'advcl':
            key1_token = token

    for token in doc_state[token_pronoun.i:]:
        if token.dep_ == 'advcl' or token.dep_ == 'relcl' or token.dep_ == 'acomp' \
            or token.dep_ == 'ROOT' or token.dep_ == 'conj' or token.dep_ == 'ccomp' \
            or token.dep_ == 'xcomp':
            key2_token = token

    if key1_token.pos_ == 'VERB' and key2_token.pos_ == 'VERB':
        relations = cn.relation(key1_token.lemma_, key2_token.lemma_)
        if 'Antonym' in relations:
            # print(doc_state)
            # print(token_candidateA)
            # print(token_candidateB)
            # print(key1_token)
            # print(key2_token)
            answer = 'B'
        elif 'Synonym' in relations:
            answer = 'A'

    elif  key1_token.pos_ == 'VERB' and key2_token.pos_ == 'ADJ':
        relations = cn.relation(key1_token.lemma_, key2_token.lemma_)
        if 'MotivatedByGoal' in relations:
            answer = 'A'
        elif 'RelatedTo' in relations:
            answer = 'B'

    return answer

def pdps_solver(statement, candidateA, candidateB, pronoun, pdps_nlp):
    #Answer metrics
    coreferenceA = dict()
    coreferenceB = dict()

    #use space pdps_nlp process to do pipline parse including tokenizer, POS, and dependency parse
    doc_state = pdps_nlp(statement)

    token_pronoun = None
    token_candidateA = None
    token_candidateB = None

    pre_candidateA, pre_candidateB = parse_candidate(candidateA, candidateB, pdps_nlp)
    pronoun = parse_pronoun(pronoun, pdps_nlp)
    token_candidateA, token_candidateB, token_pronoun = embed_candidate(pre_candidateA, pre_candidateB, pronoun, doc_state)

    #here I will use tensor in doc to compute some Coreference between pron and candidate
    Tsimilarity_A = tensor_similarity(token_candidateA, token_pronoun, doc_state)
    Tsimilarity_B = tensor_similarity(token_candidateB, token_pronoun, doc_state)

    coreferenceA['Tsimilarity'] = Tsimilarity_A
    coreferenceB['Tsimilarity'] = Tsimilarity_B

    '''
    learn from the conceptNET and try to understand the relation rule-base
    '''
    if token_candidateA.dep_ == 'nsubj' and token_candidateB.dep_ == 'dobj' and token_pronoun.dep_ == 'nsubj':
        answer = analysis_commensense(token_candidateA, token_candidateB, token_pronoun, doc_state)
    elif token_candidateA.dep_ == 'nsubj' and token_candidateB.dep_ == 'pobj' and token_pronoun.dep_ == 'nsubj':
        answer = analysis_commensense(token_candidateA, token_candidateB, token_pronoun, doc_state)
    else:
        answer = None

    if answer == None or abs(Tsimilarity_A - Tsimilarity_B) > .5:
        listy.append(Tsimilarity_A - Tsimilarity_B)
        if Tsimilarity_A >= Tsimilarity_B:
            answer = 'A'
        else:
            answer = 'B'

    return answer

if __name__ == "__main__":
    choice = str(input("What dataset do you want to evaluate (1, 2, or 3)?\n1 training data set\n2 test1 dataset\n3 test2 dataset\n4 / default all datasets\n")).strip()
    # Load in testing data
    if choice == "1":
        problems = load_data("train")
    elif choice == "2":
        problems = load_data("test1")
    elif choice == "3":
        problems = load_data("test2")
    else:
        problems = load_data("every")


    # load in spacy
    pdps_nlp =  spacy.load('en_core_web_lg')

    # Use conceptNet API for context data
    cn = conceptNet()

    problem_number = 0
    number_correct = 0

    for problem in problems:
        print(problem_number)
        #correct answer
        correct_answer = problem['correctAnswer']
        if len(correct_answer) != 1:
            correct_answer = correct_answer[0]
        #drive problem statement
        statement = problem['text']['txt1'] + ' ' + problem['text']['pron'] + ' ' + problem['text']['txt2']
        #drive two candidate and the pronoun we need understand
        candidateA = problem['answers'][0]
        candidateB = problem['answers'][1]
        pronoun = problem['text']['pron']

        problem_number += 1

        answer = pdps_solver(statement, candidateA, candidateB, pronoun, pdps_nlp)

        if answer == correct_answer:
            number_correct += 1

    print("Accuracy: ", 1.0*number_correct / problem_number)