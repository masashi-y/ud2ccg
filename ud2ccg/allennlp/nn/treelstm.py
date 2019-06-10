from overrides import overrides
from typing import List, Optional, Tuple, Union, Iterator
from itertools import zip_longest
import torch
from torch import nn
from torch.nn.modules import Dropout
from allennlp.common.checks import check_dimensions_match


class Node(object):
    def __init__(self, index: int, parent: Optional['Node'], children: List['Node']) -> None:
        self.index = index
        self.parent = parent
        self.children = children
        self.h_in = None
        self.h_up = None
        self.c_up = None
        self.h_down = None
        self.c_down = None

    def __str__(self) -> str:
        if len(self.children) == 0:
            return f'({self.index})'
        else:
            children = ', '.join(str(child) for child in self.children)
            return f'({self.index} {children})'

    def is_root(self) -> bool:
        return self.parent is None and self.index == 0


class Tree(object):
    def __init__(self, root: Node, nodes: List[Node]) -> None:
        self.root = root
        self.nodes = nodes

    def iter_topdown(self) -> Iterator[Node]:
        def rec(node: Node):
            yield node
            for child in node.children:
                yield from rec(child)
        yield from rec(self.root)

    def iter_bottomup(self) -> Iterator[Node]:
        def rec(node: Node):
            for child in node.children:
                yield from rec(child)
            yield node
        yield from rec(self.root)

    @staticmethod
    def of_list(head_indices: List[int]) -> 'Tree':
        """
        reads list of head indices and
        return Tree object representing
        dependency tree encoded in the list
        :param head_indices: heads indices where lst[0] = -1
        :return:
        """
        def rec(parent: Node) -> List[Node]:
            children = []
            for child, head in enumerate(head_indices):
                if head == parent.index:
                    child_node = Node(child, parent, [])
                    child_node.children = rec(child_node)
                    nodes[child] = child_node
                    children.append(child_node)
            return children

        nodes = [None for _ in head_indices]
        root = Node(0, None, [])
        root.children = rec(root)
        nodes[0] = root
        assert all(node is not None for node in nodes)
        return Tree(root, nodes)

    def assign_variables(self, hs: List[torch.Tensor]) -> None:
        """str(
        :param hs: a list of Variable objects whose sizes are (1, hidden_units)
        :return:
        """
        check_dimensions_match(len(hs), len(self.nodes),
                               'number of hidden vectors', 'number of nodes')
        for h, node in zip(hs, self.nodes):
            node.h_in = h

    def collect_variables(self):
        """
        :return:
        hs_up, hs_down, cs_up, cs_down: All (len(self.nodes), hidden_units)
        """
        hs_up, hs_down, cs_up, cs_down = \
            zip(*[(node.h_up, node.h_down, node.c_up, node.c_down) for node in self.nodes])
        hs_up = torch.cat(hs_up, dim=0)
        hs_down = torch.cat(hs_down, dim=0)
        cs_up = torch.cat(cs_up, dim=0)
        cs_down = torch.cat(cs_down, dim=0)
        return hs_up, hs_down, cs_up, cs_down


def split_into_batches(hs: torch.Tensor) -> List[torch.Tensor]:
    """
    :param hs: (batchsize, hidden_units)
    :return:
        a list of Variable's whose sizes are (1, hidden_units)
    """
    batchsize, _ = hs.shape
    res = torch.split(hs, 1, 0)
    return res


def make_trees(indices_or_trees: List[List[int]]) -> List[Tree]:
    """
    :param indices_or_trees: a batch of lists of head indices or Tree object
    :return:
        a list of Tree objects
    """
    trees = []
    for i_or_t in indices_or_trees:
        if not isinstance(i_or_t, Tree):
            trees.append(Tree.of_list(i_or_t))
        else:
            trees.append(i_or_t)
    return trees


def _transpose_lists_with_padding(xs, padding):
    """
    transpose a list of lists of Variable objects.
    When the lists are uneven in their length, xp.zeros are padded in the place.
    :param xs: a list of lists of Variable
    :return:
    """
    # tensor = xs[0][0]
    # padding = tensor.new_zeros(tensor.size())
    return tuple(torch.cat(x, dim=0) for x in zip_longest(*xs, fillvalue=padding))


def _pad_zero_nodes(vs: List[torch.Tensor], padding: torch.Tensor):
    if any(v is None for v in vs):
        return tuple(padding if v is None else v for v in vs)
    else:
        return vs


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W_x = torch.nn.Linear(in_size, 4 * out_size)
        self.W_h_aio = torch.nn.Linear(out_size, 3 * out_size, bias=True)
        self.W_h_f = torch.nn.Linear(out_size, out_size, bias=True)

    @overrides
    def forward(self, *cshsx):
        cs = cshsx[:len(cshsx) // 2]
        hs = cshsx[len(cshsx) // 2:-1]
        x = cshsx[-1]
        assert len(cshsx) % 2 == 1
        assert len(cs) == len(hs)

        if x is None:
            if any(c is not None for c in cs):
                base = [c for c in cs if c is not None][0]
            elif any(h is not None for h in hs):
                base = [h for h in hs if h is not None][0]
            else:
                raise ValueError('All inputs (cs, hs, x) are None.')
            x = base.new_zeros(base.size())

        x_in = self.W_x(x)
        x_aio_in, x_f_in = torch.split(x_in, [3 * self.out_size, self.out_size], dim=1)

        if len(hs) == 0:
            aio_in = x_aio_in
            a, i, o = torch.split(aio_in, self.out_size, dim=1)
            c = torch.sigmoid(i) * torch.tanh(a)
            h = torch.sigmoid(o) * torch.tanh(c)
            return c, h

        batch_size = x.size(0)
        padding = x.new_zeros(batch_size, self.out_size)
        hs = _pad_zero_nodes(hs, padding)
        cs = _pad_zero_nodes(cs, padding)

        aio_in = self.W_h_aio(sum(hs)) + x_aio_in
        h_fs_in = self.W_h_f(torch.cat(hs, dim=0)).unsqueeze(2)
        h_fs_in = torch.split(h_fs_in, batch_size, dim=0)
        # print(len(h_fs_in))
        # print(h_fs_in[0].size())
        # (batchsize, state_size, num_children)
        h_fs_in = torch.cat(h_fs_in, dim=2)
        # print(h_fs_in.size())
        f_in = h_fs_in + torch.cat([x_f_in.unsqueeze(2)] * len(hs), dim=2)

        a, i, o = torch.split(aio_in, self.out_size, dim=1)
        c = torch.cat([c.unsqueeze(2) for c in cs], dim=2) * torch.sigmoid(f_in)
        c = torch.sum(c, dim=2)
        c += torch.sigmoid(i) * torch.tanh(a)
        h = torch.sigmoid(o) * torch.tanh(c)
        return c, h


class BidirectionalTreeLSTM(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float) -> None:
        """
        :param in_size: dimensionality of input vectors
        :param out_size: dimensionality of hidden states and output vectors
        :param dropout: dropout ratio
        """
        super().__init__()
        self.in_size = in_size
        self.state_size = out_size
        if dropout == 0.0:
            self._dropout = lambda x: x
        else:
            self._dropout = Dropout(dropout)
        self._topdown_lstm = ChildSumTreeLSTM(in_size, out_size)
        self._bottomup_lstm = ChildSumTreeLSTM(in_size, out_size)

    @overrides
    def forward(self,
                xs: List[torch.Tensor],
                head_indices: List[List[int]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        :param head_indices: a list of lists of head indices
        :param xs: a list of batch size of Variable's whose sizes are (sentence length, self.in_size)
        :return:
            cs, hs, lists of the concatenation of top-down, bottom-up LSTM state variables.
            The shape of each Variable object is (sentence length, self.state_size * 2).
        """
        padding = xs[0].new_zeros(1, self.state_size)
        trees = make_trees(head_indices)
        assert len(xs) == len(trees)
        for x, tree in zip(xs, trees):
            tree.assign_variables([tensor.unsqueeze(0) for tensor in x])

        topdown_nodes = zip_longest(*[tree.iter_topdown() for tree in trees])
        bottomup_nodes = zip_longest(*[tree.iter_bottomup() for tree in trees])
        for nodes in topdown_nodes:
            # node.parent can be None
            nodes, xs, hs, cs = \
                zip(*[(node,
                       node.h_in,
                       node.parent.h_down if node.parent is not None else padding,
                       node.parent.c_down if node.parent is not None else padding)
                      for node in nodes if node is not None])
            xs = torch.cat(xs, dim=0)
            cs = torch.cat(cs, dim=0)
            hs = torch.cat(hs, dim=0)
            c_new, h_new = self._topdown_lstm(cs, hs, xs)
            c_new = self._dropout(c_new)
            h_new = self._dropout(h_new)
            for node, c_, h_ in zip(nodes, split_into_batches(c_new), split_into_batches(h_new)):
                node.c_down = c_
                node.h_down = h_

        for nodes in bottomup_nodes:
            # node.children can be empty
            nodes, xs, hs, cs = \
                zip(*[(node,
                       node.h_in,
                       [child.h_up for child in node.children],
                       [child.c_up for child in node.children])
                      for node in nodes if node is not None])
            xs = torch.cat(xs, dim=0)
            cs = _transpose_lists_with_padding(cs, padding)
            hs = _transpose_lists_with_padding(hs, padding)
            c_new, h_new = self._bottomup_lstm(*(cs + hs + (xs,)))
            c_new = self._dropout(c_new)
            h_new = self._dropout(h_new)
            for node, c_, h_ in zip(nodes, split_into_batches(c_new), split_into_batches(h_new)):
                node.c_up = c_
                node.h_up = h_

        hs_up, hs_down, cs_up, cs_down = zip(*[tree.collect_variables() for tree in trees])
        hs = [torch.cat(list(vs), dim=1) for vs in zip(hs_up, hs_down)]
        cs = [torch.cat(list(vs), dim=1) for vs in zip(cs_up, cs_down)]
        return cs, hs
