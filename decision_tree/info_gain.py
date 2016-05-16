from decision_tree import Decisiontree
from decision_tree import Data
from math import log

COLOR = 0
ROOT = 1
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
        s0 = data.num_positive() + data.num_negative()
        e0 = self.entropy(data.num_positive(), data.num_negative())
        print("Current entropy %f" %e0)
        for ai in attri_set:
            ei = 0
            for av in self.attri_list[ai]:
                # s1 = sum([av == row[ai] for row in data.positive])
                # s2 = sum([av == row[ai] for row in data.negative])
                s1 = data.num_positive_v(ai, av)
                s2 = data.num_negative_v(ai, av)
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
    attri_set = set([COLOR, ROOT, SOUND, TEXTURE, NAVEL, TOUCH])
    attri_list = {
        COLOR: [0, 1, 2],
        ROOT: [0, 1, 2],
        SOUND: [0, 1, 2],
        TEXTURE: [0, 1, 2],
        NAVEL: [0, 1, 2],
        TOUCH: [0, 1]
    }

    data_set = [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [2, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [0, 2, 2, 0, 2, 1],
        [2, 2, 2, 2, 2, 0],
        [2, 1, 0, 2, 2, 1],
        [0, 1, 0, 1, 0, 0],
        [2, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [2, 0, 0, 2, 2, 0],
        [0, 0, 1, 1, 1, 0]
    ]
    y = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
    idx = [i for i in xrange(len(y))]

    data = Data(data_set, y, idx)

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
        for child in root.children:
            q.append(child)
