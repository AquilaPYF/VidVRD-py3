import networkx as nx
from pprint import pprint as pt
import matplotlib.pyplot as plt


class TreeNode(object):
    def __init__(self, name, score, tracklet, duration):
        self.name = name
        self.score = score
        self.id = name + '_{}_{}_{}'.format(duration[0], duration[1], score)
        self.tracklet = tracklet
        self.duration = duration
        self.children = set()

    def __repr__(self):
        return self.id

    def get_all_children(self):
        children_set = set()
        if len(self.children) == 0:
            children_set.add(self)
        else:
            for each_child in self.children:
                children_set.add(each_child)
                children_set |= each_child.get_all_children()
        return children_set


class TrackTree(object):

    def __init__(self, tree_root_name, score=0., tracklet=None, duration=None):
        if duration is None:
            duration = [0, 0]
        if tracklet is None:
            tracklet = [0, 0, 0, 0]
        self.count = 0
        self.tree = TreeNode(tree_root_name, score=0., tracklet=[0, 0, 0, 0], duration=[0, 0])
        self.id = self.tree.id
        self.all_nodes = [self.tree]
        self.if_node_exist = False
        self.search_result_parent = None
        self.search_result_children = []
        self.add(TreeNode(tree_root_name, score, tracklet, duration), self.tree, trackal=True)

    def add(self, node, parent=None, trackal=True):
        if not trackal:
            self.if_node_exist = False
            self.if_node_exist_recursion(
                self.tree, node, search=False, if_del=False)

            if self.if_node_exist and parent is None:
                # print('Error: Node %s has already existed!' % node.id)
                # print('*' * 30)
                return False
            else:
                if parent is None:
                    # if parent is None, set its parent as root default
                    self.all_nodes.append(node)
                    root_children = self.tree.children
                    root_children.add(node)
                    self.tree.children = root_children
                    # print('Add node:%s sucessfully!' % node.id)
                    # print('*' * 30)
                    return True
                else:
                    # check if parent is exist
                    self.if_node_exist = False
                    self.if_node_exist_recursion(
                        self.tree, parent, search=False, if_del=False)
                    if self.if_node_exist:
                        # if has parent node
                        self.all_nodes.append(node)
                        self.add_recursion(parent.id, node, self.tree)
                        # print('Add node:%s sucessfully!' % node.id)
                        # print('*' * 30)
                        return True
                    else:
                        # print("Error: Parent node %s doesn't exist!" % parent.id)
                        # print('*' * 30)
                        return False
        else:
            # trackal add node, same level, same duration
            if not self.search(node, search_type='name'):
                # This node name doesnt exist
                return False

            node1, node2 = self.separate_node(node)
            if self.search(node1, search_type='id'):
                return False
            for each_node in self.all_nodes:
                n1s, n1e = each_node.duration
                n2s, n2e = node1.duration
                if n2s == n1e:
                    self.add(node1, each_node, False)
            if node1 in self.all_nodes:
                self.add(node2, node1, False)

    def separate_node(self, node):
        ns, ne = node.duration
        if ne - ns == 30:
            firstNode = TreeNode(
                name=node.name,
                score=node.score,
                tracklet=node.tracklet[:15],
                duration=[ns, ns + 15]
            )
            secondNode = TreeNode(
                name=node.name,
                score=node.score,
                tracklet=node.tracklet[15:],
                duration=[ns + 15, ne]
            )
            return firstNode, secondNode

    def merge2nodes(self, node1, node2):
        self.if_node_exist_recursion(
            self.tree, node1, search=True, if_del=False)
        node1_parents = self.search_result_parent
        self.if_node_exist_recursion(
            self.tree, node2, search=True, if_del=False)
        node2_parents = self.search_result_parent

        if node1.duration == node2.duration:
            return self.tree

        n1s, n1e = node1.duration
        n2s, n2e = node2.duration
        if n1s > n2e:
            # add node1 as node2 children
            # self.add(node1, node2)
            return self
        if n2s > n1e:
            # add node2 as node1 children
            # self.add(node2, node1)
            return self
        # they have overlap
        time_list = [n1s, n1e, n2s, n2e]
        time_list.sort()
        if n1s < n2s:
            start_node = node1
            end_node = node2
        else:
            start_node = node2
            end_node = node1
        sub_root = TreeNode(
            name=start_node.name,
            score=start_node.score,
            tracklet=start_node.tracklet[:time_list[1]],
            duration=time_list[0:2])
        second_node = TreeNode(
            name=start_node.name,
            score=start_node.score,
            tracklet=start_node.tracklet[time_list[1]:time_list[2]],
            duration=time_list[1:3])

        self.add(sub_root)
        self.add(end_node, sub_root)
        self.add(second_node, sub_root)
        # if n1s == 0:
        #     for each_root_child in self.tree.children:
        #         for each_start_node in each_root_child.children:
        #             self.modify(each_start_node, second_node)

        return self

    def search(self, node, search_type='id'):
        """
        print parent node id & all of son nodes
        """
        if search_type == 'id':
            self.if_node_exist = False
            self.if_node_exist_recursion(
                self.tree, node, search=True, if_del=False)
            if self.if_node_exist:
                # print("%s's parent:" % node.id)
                # pt(self.search_result_parent)
                # print("%s's children:" % node.id)
                # pt(self.search_result_children)
                # print('*' * 30)
                return True
            else:
                # print("Error: Node %s doesn't exist!" % node.id)
                # print('*' * 30)
                return False
        if search_type == 'name':
            for each_node in self.all_nodes:
                if node.name == each_node.name:
                    return True
            return False

    def get_same_duration_nodes(self, duration):
        same_level_nodes_list = []
        for each_node in self.all_nodes:
            if '{}_{}'.format(duration[0], duration[1]) in each_node:
                same_level_nodes_list.append(each_node)
        return same_level_nodes_list

    def delete(self, node):
        self.if_node_exist = False
        self.if_node_exist_recursion(
            self.tree, node, search=False, if_del=True)
        if not self.if_node_exist:
            # print("Error: Node %s doesn't exist!" % node.id)
            # print('*' * 30)
            return False
        else:
            # print('Delete node %s sucessfully!' % node.id)
            # print('*' * 30)
            return True

    def modify(self, node, new_parent=None):
        """
        change parent node
        """
        self.if_node_exist = False
        self.if_node_exist_recursion(
            self.tree, node, search=False, if_del=False)
        if not self.if_node_exist:
            # print("Error: Node %s doesn't exist!" % node.id)
            # print('*' * 30)
            return False
        else:
            if new_parent is None:
                self.if_node_exist = False
                self.if_node_exist_recursion(
                    self.tree, node, search=False, if_del=True)
                root_children = self.tree.children
                root_children.add(node)
                self.tree.children = root_children
                # print('Modify node:%s sucessfully!' % node.id)
                # print('*' * 30)
                return True
            else:
                self.if_node_exist = False
                self.if_node_exist_recursion(
                    self.tree, new_parent, search=False, if_del=False)
                if self.if_node_exist:
                    self.if_node_exist = False
                    self.if_node_exist_recursion(
                        self.tree, node, search=False, if_del=True)
                    self.add_recursion(new_parent.id, node, self.tree)
                    # print('Modify node:%s sucessfully!' % node.id)
                    # print('*' * 30)
                    return True
                else:
                    # print("Error: Parent node %s doesn't exist!" %
                    #       new_parent.id)
                    # print('*' * 30)
                    return False

    def show_tree(self):
        G = nx.Graph()
        self.to_graph_recursion(self.tree, G)
        nx.draw_networkx(G, with_labels=True, font_size=10, node_size=5)
        plt.show()

    def get_all_path(self, node=None):
        if node is None:
            node = self.tree
        if len(node.children) == 0:
            return [[]]

        paths = []
        for each_child in node.children:
            for each_child_path in self.get_all_path(each_child):
                each_path = [each_child] + each_child_path
                paths.append(each_path)

        return paths

    def get_path_score(self, path):
        path_score = 0.
        for each_node in path:
            path_score += each_node.score
        return path_score

    def get_all_nodes(self):
        all_nodes = self.all_nodes
        all_nodes.sort(key=lambda x: x.duration[0])
        return set(all_nodes)

    def to_graph_recursion(self, tree, G):
        """
        put nodes into graph
        """
        G.add_node(tree.id)
        for child in tree.children:
            G.add_nodes_from([tree.id, child.id])
            G.add_edge(tree.id, child.id)
            self.to_graph_recursion(child, G)

    def if_node_exist_recursion(self, tree, node, search, if_del):
        """
        :param tree: check whether exist node tree
        :param node: need 2 check
        :param search: when check the node, whether return parent or all of sons
        :param if_del: when check the node, whether delete it
        :return:
        """
        id = node.id
        if id == self.tree.id:
            self.if_node_exist = True
        if self.if_node_exist:
            return 1
        for child in tree.children:
            if child.id == id:
                self.if_node_exist = True
                if search is True:
                    self.search_result_parent = tree
                    for cchild in child.children:
                        self.search_result_children.append(cchild)
                elif if_del is True:
                    if node in tree.children:
                        tree.children.remove(node)
                break
            else:
                self.if_node_exist_recursion(child, node, search, if_del)

    def add_recursion(self, parent, node, tree):
        if parent == tree.id:
            tree.children.add(node)
            return 1
        for child in tree.children:
            if child.id == parent:
                children_set = child.children
                children_set.add(node)
                child.children = children_set
                break
            else:
                self.add_recursion(parent, node, child)


if __name__ == '__main__':
    T = TrackTree(tree_root_name='adult#0',
                  score=0.111,
                  tracklet=[1, 1, 1, 1],
                  duration=[0, 30])
    A = TreeNode('adult#0', 0.1, [2, 2, 2, 2], [0, 30])
    B = TreeNode('adult#0', 0.2, [3, 3, 3, 3], [0, 30])
    C = TreeNode('adult#0', 0.3, [4, 4, 4, 4], [15, 45])
    D = TreeNode('adult#0', 0.4, [5, 5, 5, 5], [30, 60])
    E = TreeNode('adult#0', 0.5, [6, 6, 6, 6], [30, 60])

    T.add(A)
    T.add(B)
    T.add(C)
    T.add(C)
    T.add(D)
    # for each in T.get_all_nodes():
    #     print(each.id, each.children)
    # T.show_tree()

    # print(T.tree)
    for each_path in T.get_all_path():
        for each_node in each_path:
            print(each_node.id, end='->')
        print(T.get_path_score(each_path), end='')
        print()
