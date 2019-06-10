
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from allennlp.common import Registrable
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from ud2ccg.allennlp.nn.treelstm import BidirectionalTreeLSTM, ChildSumTreeLSTM


class BidirectionalTreeLSTMEncoder(Registrable, nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float=0.5) -> None:
        super().__init__()
        assert out_size % 2 == 0
        self.encoder = BidirectionalTreeLSTM(in_size, out_size // 2, dropout)

    def forward(self, inputs: torch.Tensor, head_indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (batchsize, sequence length + 1, unit size)
        :param head_indices: (batchsize, sequence length + 1)
        :param mask: (batchsize, sequence length + 1)
        "sequence length + 1" is for a root token.
        :return:
        """
        lengths = get_lengths_from_binary_sequence_mask(mask)
        inputs = [x[:length] for x, length in zip(inputs, lengths)]
        split_head_indices = [list(instance_heads[:length]) for instance_heads, length
                              in zip(head_indices, lengths)]
        _, hs = self.encoder(inputs, split_head_indices)
        results = pad_sequence(hs, batch_first=True)
        return results

    def get_output_dim(self):
        return self.encoder.state_size

    def get_input_dim(self):
        return self.encoder.in_size


def test():
    encoder = BidirectionalTreeLSTMEncoder(10, 20)
    inputs = torch.ones(5, 6, 10)
    head_indices = [torch.zeros(5, dtype=torch.int).unsqueeze(-1) + i - 1 for i in range(6)]
    head_indices = torch.cat(head_indices, dim=1)
    print(head_indices)
    mask = torch.ones(5, 6, dtype=torch.int)
    res = encoder(inputs, head_indices, mask)
    print(res)
    print(res[0].size())
    print(res.size())
    print(len(res))

