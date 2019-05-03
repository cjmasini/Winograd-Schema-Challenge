import xml.etree.ElementTree as etree

class schema:
    def __init__(self, q, cr, c1, c2, a):
        self.question = q.strip()
        self.coreference = cr.strip()
        self.choice1 = c1.strip()
        self.choice2 = c2.strip()
        self.answer = a.strip()

    def __str__(self):
        str = ""
        str += "Question: " + self.question
        str += "\nCoreference: " + self.coreference
        str += "\nChoice 1: " + self.choice1
        str += "\nChoice 2: " + self.choice2
        str += "\nAnswer: " + self.answer
        return str

    def to_dict(self):
        dict = {'text': {}}
        dict['text']['txt1'] = self.question[:self.question.index(" " + self.coreference)].strip()
        dict['text']['pron'] = self.coreference
        dict['text']['txt2'] = self.question[self.question.index(" " + self.coreference) + len(self.coreference) + 1:].strip()
        dict['answers'] = [self.choice1, self.choice2]
        dict['correctAnswer'] = 'A' if self.answer == self.choice1 else 'B'
        return dict

def load_data(choice):
    print("Loading {} dataset".format(choice))
    data = []
    if choice == "train":
        with open("./Winograd/train/train.c.txt") as f:
            content = f.readlines()
        i = 0
        while i < len(content):
            q = schema(content[i], content[i+1], content[i+2].split(",")[0], content[i+2].split(",")[1], content[i+3])
            data.append(q.to_dict())
            i += 5
    elif choice == "test1":
        with open("./Winograd/test1/test1.c.txt") as f:
            content = f.readlines()
        i = 0
        while i < len(content):
            q = schema(content[i], content[i+1], content[i+2].split(",")[0], content[i+2].split(",")[1], content[i+3])
            data.append(q.to_dict())
            i += 5
    elif choice == "test2":
        tree = etree.parse('./Winograd/test2/WSCollection.xml')
        root = tree.getroot()
        data = list()
        original_problems = root.getchildren()

        for original_problem in original_problems:
            problem = dict()
            for information in original_problem.getchildren():
                if information.tag == 'answers':
                    answers = information.getchildren()
                    answer_list = list()
                    for answer in answers:
                        answer_list.append(answer.text.strip())
                    problem['answers'] = answer_list
                elif information.tag == 'text':
                    texts = information.getchildren()
                    text_dict = dict()
                    for text1 in texts:
                        text_dict[text1.tag] = text1.text.replace('\n', ' ').strip()
                    problem['text'] = text_dict
                elif information.tag == 'quote':
                    pass
                else:
                    problem[information.tag] = information.text.replace(' ', '')
            data.append(problem)
    else:
        tree = etree.parse('./Winograd/test2/WSCollection.xml')
        root = tree.getroot()
        data = list()
        original_problems = root.getchildren()

        for original_problem in original_problems:
            problem = dict()
            for information in original_problem.getchildren():
                if information.tag == 'answers':
                    answers = information.getchildren()
                    answer_list = list()
                    for answer in answers:
                        answer_list.append(answer.text.strip())
                    problem['answers'] = answer_list
                elif information.tag == 'text':
                    texts = information.getchildren()
                    text_dict = dict()
                    for text1 in texts:
                        text_dict[text1.tag] = text1.text.replace('\n', ' ').strip()
                    problem['text'] = text_dict
                elif information.tag == 'quote':
                    pass
                else:
                    problem[information.tag] = information.text.replace(' ', '')
            data.append(problem)
        with open("./Winograd/test1/test1.c.txt") as f:
            content = f.readlines()
        i = 0
        while i < len(content):
            q = schema(content[i], content[i+1], content[i+2].split(",")[0], content[i+2].split(",")[1], content[i+3])
            data.append(q.to_dict())
            i += 5
        with open("./Winograd/train/train.c.txt") as f:
            content = f.readlines()
        i = 0
        while i < len(content):
            q = schema(content[i], content[i+1], content[i+2].split(",")[0], content[i+2].split(",")[1], content[i+3])
            data.append(q.to_dict())
            i += 5

    return data