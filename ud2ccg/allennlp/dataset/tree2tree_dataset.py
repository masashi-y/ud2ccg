from typing import Dict, List, Any
import logging
import numpy
from overrides import overrides
import json
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from ud2ccg.allennlp.data.fields.int_array_field import IntArrayField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("tree2tree_dataset")
class Tree2TreeDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ud_tag_token_indexers: Dict[str, TokenIndexer] = None,
                 ud_label_token_indexers: Dict[str, TokenIndexer] = None,
                 use_ancestor_field: bool = False,
                 use_path_pattern_field: bool = False) -> None:
        super().__init__(lazy)
        self.use_ancestor_field = use_ancestor_field
        self.use_path_pattern_field = use_path_pattern_field
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._ud_tag_token_indexers = ud_tag_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._ud_label_token_indexers = ud_label_token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            logger.info('Reading instances from lines in file at: %s', file_path)

            for (words, _), source, target in json.load(data_file):
                (ud_head_indices, ud_tags, ud_labels) = source
                (head_tags, head_indices) = target
                yield self.text_to_instance(words=words,
                                            ud_head_indices=ud_head_indices[1:],
                                            ud_tags=ud_tags[1:],
                                            ud_labels=ud_labels[1:],
                                            head_tags=head_tags,
                                            head_indices=head_indices)

    @overrides
    def text_to_instance(self,
                         words: List[str],
                         ud_head_indices: List[int],
                         ud_tags: List[str],
                         ud_labels: List[str],
                         head_tags: List[str] = None,
                         head_indices: List[int] = None,
                         metadata: Dict[str, Any] = None) -> Instance:
        # pylint: disable=arguments-differ
        token_field = TextField(list(map(Token, words)), self._token_indexers)
        ud_head_index_field = IntArrayField(numpy.array([-1] + ud_head_indices, 'i'))
        ud_tag_field = TextField(list(map(Token, ud_tags)), self._ud_tag_token_indexers)
        ud_label_field = TextField(list(map(Token, ud_labels)), self._ud_label_token_indexers)
        metadata = metadata or {}
        metadata['words'] = words
        metadata = MetadataField(metadata)
        fields = {
            'words': token_field,
            'metadata': metadata,
            'ud_head_index_field': ud_head_index_field,
            'ud_tag_field': ud_tag_field,
            'ud_label_field': ud_label_field,
        }
        if self.use_ancestor_field:
            ancestor_field = SequenceLabelField(
                get_least_common_ancestor(ud_head_indices), token_field, label_namespace='ancestors')
            fields['ancestor_field'] = ancestor_field
        if self.use_path_pattern_field:
            path_pattern_field = SequenceLabelField(
                dependency_path_pattern(ud_head_indices), token_field, label_namespace='path_pattern')
            fields['path_pattern_field'] = path_pattern_field

        if head_tags is not None and head_indices is not None:
            fields['head_tags'] = SequenceLabelField(
                head_tags, token_field, label_namespace='head_tags')
            fields['head_indices'] = SequenceLabelField(
                head_indices, token_field, label_namespace='head_indices')
        return Instance(fields)
