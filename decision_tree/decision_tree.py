COLOR = 0
ROOT  = 1
SOUND = 2
TEXTURE = 3
NAVEL = 4
TOUCH = 5


from aenum import Enum
from copy import deepcopy


class Attribute(Enum):
    COLOR = 0
    ROOT  = 1
    SOUND = 2
    TEXTURE = 3
    NAVEL = 4
    TOUCH = 5


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


class DataSet(object):
    """
    Use to manage data set and extract data set.
    """
    def __init__(self):
        self.positive = []
        self.negative = []

        # self.readdata()

    def readdata(self):
        self.positive = [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [2, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 0, 0, 1, 0]
        ]
        self.negative = [
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

    def empty(self):
        """
        Check whether data set is empty.
        :return: bool
        """
        if len(self.positive) == 0 and len(self.negative) == 0:
            return True
        return False

    def filterdata(self, ai, val):
        """
        Extract row data and return new data set
        :param ai: int
        :param val: int
        :return: DataSet()
        """
        data_v = DataSet()
        data_v.positive = [x for x in self.positive if x[ai] == val]
        data_v.negative = [x for x in self.negative if x[ai] == val]
        return data_v

    def mark_most(self):
        if self.empty():
            return None
        if len(self.positive) >= len(self.negative):
            return  True
        return False


class Decisiontree(object):
    """
    Decision Tree base class
    """
    def __init__(self, attri_list):
        self.attri_list = attri_list

    def find_best(self, data, attri_set):
        best = attri_set.pop()
        return best

    def tree_gen(self, data, attri_set):
        """
        :param data_set: DataSet()
        :param attri_set: set()
        :return:
        """
        newNode = Node()
        # If data set is already classified.
        if len(data.positive) == 0:
            newNode.set_leaf(False)
            return newNode
        elif len(data.negative) == 0:
            newNode.set_leaf(True)
            return newNode

        # If attribute set is empty.
        if not attri_set:
            type = data.mark_most()
            newNode.set_leaf(type)
            return newNode

        # Find a best decision attribute.
        choice = self.find_best(data, attri_set)
        if choice == -1:
            print "error"
            return None
        print "best choice:", Attribute(choice)
        newNode.attri = Attribute(choice)

        new_attri_set = deepcopy(attri_set)
        new_attri_set.remove(choice)

        for val in self.attri_list[choice]:
            data_v = data.filterdata(choice, val)
            # print data_v.positive, data_v.negative
            if data_v.empty():
                childNode = Node()
                childNode.set_leaf(data.mark_most())  # set parent's most
                newNode.children.append(childNode)
            else:
                childNode = self.tree_gen(data_v, new_attri_set)
                newNode.children.append(childNode)

        return newNode


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
    d = Decisiontree(attri_list)
    root = d.tree_gen(data, attri_set)

    print root