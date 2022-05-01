from sklearn.ensemble import RandomForestClassifier
import math


class Domain:
    def __init__(self, _name, _label, _length, _num, _entropy):
        self.name = _name
        self.label = _label
        self.length = _length
        self.num = _num
        self.entropy = _entropy

    def returnData(self):
        return [self.length, self.num, self.entropy]

    def returnLabel(self):
        if self.label == "dga":
            return 1
        return 0


def cal_num(str):
    num = 0
    for i in str:
        if i.isdigit():
            num += 1
    return num


def cal_seg(str):
    num = 0
    for i in str:
        if i == '.':
            num += 1
    return num


def cal_entropy(str):
    h = 0.0
    sumLetter = 0
    letter = [0] * 26
    str = str.lower()
    for i in range(len(str)):
        if str[i].isalpha():
            letter[ord(str[i]) - ord('a')] += 1
            sumLetter += 1
    for i in range(26):
        p = 1.0 * letter[i] / sumLetter
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


def initData(filename, domainlist):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            if len(tokens) > 1:
                label = tokens[1]
            else:
                label = "?"
            length = len(name)
            num = cal_num(name)
            entropy = cal_entropy(name)
            # seg = cal_seg(name)
            domainlist.append(Domain(name, label, length, num, entropy))



domainlist1 = []
initData("train.txt", domainlist1)
featureMatrix = []
labelList = []
for item in domainlist1:
    featureMatrix.append(item.returnData())
    labelList.append(item.returnLabel())

clf = RandomForestClassifier(random_state=0)
clf.fit(featureMatrix, labelList)

domainlist2 = []
initData("test.txt", domainlist2)
with open("result.txt", "w") as f:
    for i in domainlist2:
        f.write(i.name)
        f.write(",")
        if clf.predict([i.returnData()])[0] == 0:
            f.write("notdga")
        else:
            f.write("dga")
        f.write("\n")

