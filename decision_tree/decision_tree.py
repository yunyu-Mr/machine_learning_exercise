COLOR = 0
ROOT = 1
SOUND = 2
TEXTURE = 3
NAVEL = 4
TOUCH = 5
DENSITY = 6
SUGAR = 7


from aenum import Enum
from copy import deepcopy
import numpy as np

class Attribute(Enum):
    COLOR = 0
    ROOT = 1
    SOUND = 2
    TEXTURE = 3
    NAVEL = 4
    TOUCH = 5
    DENSITY = 6
    SUGAR = 7


class Node(object):
    def __init__(self):
        self.attri = None
        self.isLeaf = False
        self.decision = False
        self.parent = None
        self.children = []

    def set_leaf(self, positive):
        """
        :param positive: bool
        :return:
        """
        self.isLeaf = True
        self.decision = positive


# class DataSet(object):
#     """
#     Use to manage data set and extract data set.
#     """
#     def __init__(self):
#         self.positive = []
#         self.negative = []
#
#     def readdata(self):
#         self.positive = [
#             [0, 0, 0, 0, 0, 0],
#             [1, 0, 1, 0, 0, 0],
#             [1, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0],
#             [2, 0, 0, 0, 1, 0],
#             [0, 1, 0, 0, 1, 1],
#             [1, 1, 0, 1, 1, 1],
#             [1, 1, 0, 0, 1, 0]
#         ]
#         self.negative = [
#             [1, 1, 1, 1, 1, 0],
#             [0, 2, 2, 0, 2, 1],
#             [2, 2, 2, 2, 2, 0],
#             [2, 1, 0, 2, 2, 1],
#             [0, 1, 0, 1, 0, 0],
#             [2, 1, 1, 1, 0, 0],
#             [1, 1, 0, 0, 1, 1],
#             [2, 0, 0, 2, 2, 0],
#             [0, 0, 1, 1, 1, 0]
#         ]
#
#     def empty(self):
#         """
#         Check whether data set is empty.
#         :return: bool
#         """
#         if len(self.positive) == 0 and len(self.negative) == 0:
#             return True
#         return False
#
#     def filterdata(self, ai, val):
#         """
#         Extract row data and return new data set
#         :param ai: int
#         :param val: int
#         :return: DataSet()
#         """
#         data_v = DataSet()
#         data_v.positive = [x for x in self.positive if x[ai] == val]
#         data_v.negative = [x for x in self.negative if x[ai] == val]
#         return data_v
#
#     def mark_most(self):
#         if self.empty():
#             return None
#         if len(self.positive) >= len(self.negative):
#             return  True
#         return False


class DataB(object):
    def __init__(self, data_set, y, idx):
        """
        :param data_set: List(List(int))
        :param y: List(int)
        :param idx: List(int)
        """
        self.data = data_set
        self.y = y
        self.idx = idx

    def filter(self, ai, val):
        """
        :param ai: int
        :param val: int
        :return: DataB()
        """
        filter_idx = [x for x in self.idx if self.data[x][ai] == val]
        return DataB(self.data, self.y, filter_idx)

    def empty(self):
        if len(self.idx) == 0:
            return True
        return False

    def is_positive(self):
        if len(self.idx) == sum([self.y[i] for i in self.idx]):
            return True
        return False

    def is_negative(self):
        if sum([self.y[i] for i in self.idx]) == 0:
            return True
        return False

    def mark_most(self):
        if self.empty():
            return None
        num_pos = sum([self.y[i] for i in self.idx])
        num_neg = len(self.idx) - num_pos
        print("num_pos: %d, num_neg: %d" %(num_pos, num_neg))
        if num_pos >= num_neg:
            return True
        return False


class Decisiontree(object):
    """
    Decision Tree base class
    """
    def __init__(self, attri_list):
        self.attri_list = attri_list

    def find_best(self, data, attri_set):
        best = list(attri_set)[0]
        return best

    def tree_gen(self, data, attri_set):
        """
        :param data_set: Data()
        :param attri_set: set()
        :return:
        """
        newNode = Node()
        # If data set is already classified.
        if data.is_positive():
            newNode.set_leaf(True)
            return newNode
        elif data.is_negative():
            newNode.set_leaf(False)
            return newNode

        # If attribute set is empty.
        if not attri_set:
            type = data.mark_most()
            newNode.set_leaf(type)
            return newNode

        # Find a best decision attribute.
        choice, midpoint = self.find_best(data, attri_set)
        if choice == -1:
            print "error"
            return None
        print "best choice:", Attribute(choice), midpoint
        newNode.attri = Attribute(choice)

        new_attri_set = deepcopy(attri_set)
        new_attri_set.remove(choice)

        for val in self.attri_list[choice]:
            data_v = data.filter(choice, val, midpoint=midpoint)
            if data_v.empty():
                childNode = Node()
                childNode.set_leaf(data.mark_most())  # set parent's most
                newNode.children.append(childNode)
            else:
                childNode = self.tree_gen(data_v, new_attri_set)
                newNode.children.append(childNode)

        return newNode


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

    data_set = np.array([
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
    ], np.float16)
    y = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
    idx = [i for i in xrange(len(y))]

    data = DataB(data_set, y, idx)

    d = Decisiontree(attri_list)
    root = d.tree_gen(data, attri_set)

    print root