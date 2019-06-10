from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import allennlp.predictors.predictor as predictor
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from utils import denormalize


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Predictor.register('tree2tree-predictor')
class Tree2treePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        raise NotImplementedError('no support for inference on a raw sentence.')

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(words=json_dict['words'],
                                                     ud_head_indices=json_dict['heads'],
                                                     ud_tags=json_dict['tags'],
                                                     ud_labels=json_dict['head_labels'],
                                                     metadata=json_dict.get('metadata', None))

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        [result] = self._make_json([outputs])
        return result

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return self._make_json(outputs)

    def _make_json(self, output_dicts: List[Dict[str, Any]]) -> List[JsonDict]:
        categories = self._model.vocab.get_index_to_token_vocabulary('head_tags')
        categories = [token for _, token in sorted(categories.items())]
        categories, paddings = categories[2:], categories[:2]
        assert all(padding in [DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN] for padding in paddings)

        for output_dict in output_dicts:
            output_dict['categories'] = categories
            head_tags = output_dict["head_tags"]
            assert head_tags.shape[-1] == len(categories)
            heads = output_dict["heads"]
            output_dict["head_tags"] = head_tags.flatten().astype(float).tolist()
            output_dict["heads"] = heads.flatten().astype(float).tolist()
            output_dict["head_tags_shape"] = list(head_tags.shape)
            output_dict["heads_shape"] = list(heads.shape)
            output_dict["words"] = ' '.join(output_dict["words"])
            output_dict.pop("loss")
            output_dict.pop("mask")
        return output_dicts


@Predictor.register('tree2tree-switchboard-predictor')
class SwitchboardTree2treePredictor(Tree2treePredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        raise NotImplementedError('no support for inference on a raw sentence.')

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(words=json_dict['words'],
                                                     ud_head_indices=json_dict['heads'],
                                                     ud_tags=json_dict['tags'],
                                                     ud_labels=json_dict['head_labels'],
                                                     metadata=json_dict.get('metadata', None))

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        [result] = self._make_json([outputs])
        return result

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return self._make_json(outputs)

    def _make_json(self, output_dicts: List[Dict[str, Any]]) -> List[JsonDict]:
        categories = self._model.vocab.get_index_to_token_vocabulary('head_tags')
        categories = [token for _, token in sorted(categories.items())]
        categories, paddings = categories[2:], categories[:2]
        assert all(padding in [DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN] for padding in paddings)

        categories.append('X')
        category_size = len(categories)
        for output_dict in output_dicts:
            fluent_sentence = output_dict['words']
            original_sentence = output_dict['original']
            head_tags = output_dict['head_tags']
            heads = output_dict['heads']
            edit_info = output_dict['edits']
            if not output_dict['contain_disfluency']:
                assert len(fluent_sentence) == len(original_sentence)
                new_head_tags = numpy.ones((len(original_sentence), category_size), 'f') * -numpy.inf
                new_head_tags[:, :-1] = head_tags
                head_tags = new_head_tags
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    predicted_head_tags = numpy.argmax(head_tags, axis=1)
                    predicted_heads = numpy.argmax(heads, axis=1)
                    for i, (word, edit, cat, head) in enumerate(zip(
                            fluent_sentence, edit_info, predicted_head_tags, predicted_heads), 1):
                        edit = "1" if edit else "0"
                        logger.debug(f'{i}\t{word}\t{head}\t{edit}\t{categories[cat]}')
                    logger.debug('')

                disfl_indices = [i for i, edit in enumerate(edit_info) if edit]
                fluent_indices = [i for i, edit in enumerate(edit_info) if not edit]
                assert len(fluent_indices) == len(fluent_sentence), str(fluent_sentence)

                new_head_tags = numpy.ones((len(original_sentence), category_size), 'f') * -numpy.inf
                new_head_tags[fluent_indices, :-1] = head_tags
                new_head_tags[disfl_indices, -1] = 0.0  # log prob 0 for category 'X'

                new_heads = numpy.ones((len(original_sentence), len(original_sentence) + 1), 'f') * -numpy.inf
                new_heads[numpy.ix_(fluent_indices, [0] + [v + 1 for v in fluent_indices])] = heads
                new_heads[numpy.ix_(fluent_indices, [v + 1 for v in disfl_indices])] = -numpy.inf
                new_heads[disfl_indices, disfl_indices] = 0.0  # attach disfluencies to previous word
                if 0 in disfl_indices:
                    # set this token's head to the last disfluent token before it.
                    clean0 = fluent_indices[0]
                    disfl0 = [i for i in disfl_indices if i <= clean0][-1]
                    new_heads[clean0, disfl0 + 1] = 0.0
                    new_heads[clean0, 0] = - numpy.inf
                head_tags = new_head_tags
                heads = new_heads

                if logger.isEnabledFor(logging.DEBUG):
                    predicted_head_tags = numpy.argmax(head_tags, axis=1)
                    predicted_heads = numpy.argmax(heads, axis=1)
                    for i, (word, edit, cat, head) in enumerate(zip(
                            original_sentence, edit_info, predicted_head_tags, predicted_heads), 1):
                        edit = "1" if edit else "0"
                        logger.debug(f'{i}\t{word}\t{head}\t{edit}\t{categories[cat]}')
                    logger.debug('')

            assert head_tags.shape[-1] == category_size
            output_dict['categories'] = categories
            output_dict["head_tags"] = head_tags.flatten().astype(float).tolist()
            output_dict["heads"] = heads.flatten().astype(float).tolist()
            output_dict["head_tags_shape"] = list(head_tags.shape)
            output_dict["heads_shape"] = list(heads.shape)
            output_dict["words"] = ' '.join(original_sentence)
            output_dict.pop("loss")
            output_dict.pop("mask")
        return output_dicts


predictor.DEFAULT_PREDICTORS['tree2tree_bitreelstm'] = 'tree2tree-predictor'
