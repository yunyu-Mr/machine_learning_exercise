from decision_tree import Decisiontree
from decision_tree import DataB
from math import log
import numpy as np
import preprocess

COLOR = 0
ROOT = 1
SOUND = 2
TEXTURE = 3
NAVEL = 4
TOUCH = 5
DENSITY = 6
SUGAR = 7

discrete_set = set([COLOR, ROOT, SOUND, TEXTURE, NAVEL, TOUCH])
continuous_set = set([SUGAR, DENSITY])


class Data(DataB):
    """
    Data set access interface.
    """
    def filter(self, ai, val, midpoint=None):
        """
        Override filter.
        :param ai: int
        :param val: int
        :param midpoint: float
        :return: Data()
        """
        if midpoint is not None:
            if val == 0:
                filter_idx = [x for x in self.idx if self.data[x][ai] <= midpoint]
            elif val == 1:
                filter_idx = [x for x in self.idx if self.data[x][ai] > midpoint]
        else:
            filter_idx = [x for x in self.idx if self.data[x][ai] == val]
        return Data(self.data, self.y, filter_idx)

    def get_filter_idx(self, ai, av, is_cont=False):
        """
        :param ai: int
        :param av: int
        :param is_cont: bool
        :return: List(int)
        """
        if is_cont:
            return [x for x in self.idx if self.data[x][ai] < av]
        return [x for x in self.idx if self.data[x][ai] == av]

    def num_positive(self):
        """
        :return: int
        """
        return sum([self.y[i] for i in self.idx])

    def num_positive_v(self, ai, av, is_cont=False):
        """
        :param ai: int
        :param av: int
        :param is_cont: bool
        :return: int
        """
        filter_list = self.get_filter_idx(ai, av, is_cont)
        return sum([self.y[i] for i in filter_list])

    def num_negative(self):
        """
        :return: int
        """
        return len(self.idx) - sum([self.y[i] for i in self.idx])

    def num_negative_v(self, ai, av, is_cont=False):
        """
        :param ai: int
        :param av: int
        :return: int
        """
        filter_list = self.get_filter_idx(ai, av, is_cont)
        return len(filter_list) - sum([self.y[i] for i in filter_list])


class DecisionTreeInfoGain(Decisiontree):
    """
    Decision Tree calculate by Information Gain.
    Mainly override the find_best method.
    """
    def __init__(self, attri_list, midval=None):
        """
        :param attri_list: Dict()
        :param midval: Dict()
        :return:
        """
        super(self.__class__, self).__init__(attri_list)
        self.midval = midval

    def find_best(self, data, attri_set):
        """
        Override find_best method
        :param data: Data()
        :param attri_set: Set(int)
        :return: int
        """
        # Best choice with max info gain.
        best = -1
        # Calculate max info gain.
        max_gain = 0
        # If continuous attribute, find a best mid break point.
        point = None

        # Entropy of original.
        s0 = data.num_positive() + data.num_negative()
        e0 = self.entropy(data.num_positive(), data.num_negative())
        print("Current entropy %f" %e0)

        # Calculate entropy of different branches.
        for ai in attri_set:
            ei = 0
            # Check whether ai is continuous attribute.
            if ai in discrete_set:
                # For discrete attributes.
                for av in self.attri_list[ai]:
                    s1 = data.num_positive_v(ai, av)
                    s2 = data.num_negative_v(ai, av)
                    if s1 + s2 == 0:
                        continue
                    ei += (s1+s2)/float(s0) * self.entropy(s1, s2)
                info_gain = e0 - ei
                if info_gain > max_gain:
                    max_gain = info_gain
                    best = ai
            else:
                # For continuous attributes.
                sp_t = data.num_positive()
                sn_t = data.num_negative()
                for av in self.midval[ai]:
                    ei = 0
                    sp_l = data.num_positive_v(ai, av, True)
                    sn_l = data.num_negative_v(ai, av, True)
                    if sp_l + sn_l != 0:
                        ei += (sp_l+sn_l)/float(s0) * self.entropy(sp_l, sn_l)
                    sp_r = sp_t - sp_l
                    sn_r = sn_t - sn_l
                    if sp_r + sn_r != 0:
                        ei += (sp_r+sn_r)/float(s0) * self.entropy(sp_r, sn_r)
                    info_gain = e0 - ei
                    if info_gain > max_gain:
                        max_gain = info_gain
                        best = ai
                        # Remember to record mid point.
                        point = av
        if point is not None:
            print("Mid Point: %f" % point)
        return best, point

    @staticmethod
    def entropy(s1, s2):
        """
        Entropy calculator.
        :param s1: int
        :param s2: int
        :return: float
        """
        p1 = float(s1) / (s1 + s2)
        p2 = float(s2) / (s1 + s2)
        if p1 == 0 or p2 == 0:
            return 0
        try:
            # Only two label: T or F.
            ent = (- p1 * log(p1, 2) - p2 * log(p2, 2))
        except:
            ent = 0
        return ent


if __name__ == '__main__':
    # Combine discrete and continuous attributions.
    attri_set = discrete_set.union(continuous_set)
    # All possible branches.
    attri_list = {
        COLOR: [0, 1, 2],
        ROOT: [0, 1, 2],
        SOUND: [0, 1, 2],
        TEXTURE: [0, 1, 2],
        NAVEL: [0, 1, 2],
        TOUCH: [0, 1],
        DENSITY: [0, 1],
        SUGAR: [0, 1]
    }

    # Load data from text file.
    data_file = "data_set3.0.txt"
    data_set = np.loadtxt(data_file, dtype=np.float16, delimiter=',')

    # Process continuous attribution val.
    midval = dict()
    for ai in continuous_set:
        val_list = preprocess.cont2mid(data_set, ai)
        midval[ai] = val_list

    # Data label.
    y = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
    # Data index.
    idx = [i for i in xrange(len(y))]

    # Create data object to manage data.
    data = Data(data_set, y, idx)

    # Create decision tree object
    dt = DecisionTreeInfoGain(attri_list, midval=midval)
    # Generate decision tree.
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
