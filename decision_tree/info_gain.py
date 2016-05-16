from decision_tree import Decisiontree
from decision_tree import DataSet
from math import log

COLOR = 0
ROOT  = 1
SOUND = 2
TEXTURE = 3
NAVEL = 4
TOUCH = 5


class DecisionTreeInfoGain(Decisiontree):
    """
    Decision Tree calculate by Information Gain
    """
    def find_best(self, data, attri_set):
        """
        Override find_best method
        :param attri_set:
        :return: int
        """
        best = -1
        max_gain = 0
        # Entropy of original
        s0 = len(data.positive) + len(data.negative)
        e0 = self.entropy(len(data.positive), len(data.negative))
        print("Current entropy %f" %e0)
        for ai in attri_set:
            ei = 0
            for av in self.attri_list[ai]:
                s1 = sum([av == row[ai] for row in data.positive])
                s2 = sum([av == row[ai] for row in data.negative])
                if s1 + s2 == 0:
                    continue
                ei += (s1+s2)/float(s0) * self.entropy(s1, s2)
            info_gain = e0 - ei
            if info_gain > max_gain:
                max_gain = info_gain
                best = ai
        return best

    def entropy(self, s1, s2):
        """
        :param s1: int
        :param s2: int
        :return: float
        """
        p1 = float(s1) / (s1 + s2)
        p2 = float(s2) / (s1 + s2)
        if p1 == 0 or p2 == 0:
            return 0
        try:
            ent = (- p1 * log(p1, 2) - p2 * log(p2, 2))
        except:
            ent = 0
        return ent

if __name__ == '__main__':
    data = DataSet()
    data.readdata()

    attri_set = set([COLOR, ROOT, SOUND, TEXTURE, NAVEL, TOUCH])
    attri_list = {
        COLOR: [0, 1, 2],
        ROOT: [0, 1, 2],
        SOUND: [0, 1, 2],
        TEXTURE: [0, 1, 2],
        NAVEL: [0, 1, 2],
        TOUCH: [0, 1]
    }

    dt = DecisionTreeInfoGain(attri_list)
    root = dt.tree_gen(data, attri_set)

    print("Travel tree, breath first search")
    q = [root]
    while len(q) > 0:
        root = q.pop(0)
        if root.isLeaf:
            print("Good or Bad", root.decision)
        else:
            print("Choice: ", root.attri)
            # print root.children
        for child in root.children:
            q.append(child)
