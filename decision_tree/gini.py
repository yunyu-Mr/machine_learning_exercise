from decision_tree import DataB
from decision_tree import Decisiontree
import numpy as np

COLOR = 0
ROOT = 1
SOUND = 2
TEXTURE = 3
NAVEL = 4
TOUCH = 5
DENSITY = 6
SUGAR = 7


class Data(DataB):
    def filter(self, ai, val, midpoint=None):
        filter_idx = [x for x in self.idx if self.data[x][ai] == val]
        return Data(self.data, self.y, filter_idx)

    def filter_count(self, ai, av):
        """
        :param ai: int
        :param av: int
        :return: (int, List(int))
        """
        filter_idx = [x for x in self.idx if self.data[x][ai] == av]
        l = [self.y[i] for i in filter_idx]
        return len(filter_idx), [l.count(0), l.count(1)]

    def size(self):
        return len(self.idx)


class DecisionTreeGini(Decisiontree):
    def find_best(self, data, attri_set):
        """
        Override find_best method
        :param data: Data()
        :param attri_set: Set(int)
        :return: int
        """
        # Best choice with max info gain.
        best = -1
        # Gini_index
        min_gini = 1
        for ai in attri_set:
            gini_idx = self.gini_index(data, ai)
            print("Gini of %d is %f" %(ai, gini_idx))
            if gini_idx < min_gini:
                min_gini = gini_idx
                best = ai
        print("Minimum Gini: %f" % min_gini)
        return best, None

    def gini_index(self, data, ai):
        """
        :param data: Data()
        :param ai: int
        :return: float
        """
        gini_idx = 0
        dsize = data.size()
        for av in self.attri_list[ai]:
            dsize_v, cnt = data.filter_count(ai, av)
            gini_idx += (float(dsize_v) / dsize) * self.gini(cnt)
        return gini_idx

    @staticmethod
    def gini(cnt):
        """
        :param cnt: List(int)
        :return: float
        """
        s = sum(cnt)
        gi = 1
        if s == 0:
            return gi
        # print cnt
        for c in cnt:
            gi -= (float(c) / s) ** 2
        # print gi
        return gi

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

    # Load data set
    data_set = np.loadtxt("data_set2.0_train.txt", dtype=np.int8, delimiter=',')

    # Data label.
    y = [1,1,1,1,1,0,0,0,0,0]
    # Data index.
    idx = [i for i in xrange(len(y))]

    # Create data object to manage data.
    data = Data(data_set, y, idx)

    # Create decision tree based on gini index.
    dt = DecisionTreeGini(attri_list)
    root = dt.tree_gen(data, attri_set)

    # Print the decision tree (BFS).
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