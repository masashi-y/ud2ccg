(import 'supertagger.jsonnet') + {
  dataset_reader+: {
    type: 'finetune_supertagging_dataset',
    lazy: true,
    // "noisy_weight": 0.4,
    ccgbank_ratio: 0.1,  // genia train: 4467 sentences
    // tritrain_ratio: 0.5,
    auxiliary_ratio: 1,
    token_indexers: super.token_indexers,
  },
  train_data_path: {
    ccgbank: 'http://cl.naist.jp/~masashi-y/resources/tagger_data/traindata.json',
    auxiliary: '/home/cl/masashi-y/ud2ccg/py/models/tree2tree_models/tree2tree_input_variational_elmo/genia_with_constraints_temperature0.5_0.5/all_without_fail.json',
  },
  validation_data_path: '/home/cl/masashi-y/ud2ccg/genia/GENIA1000_eval.jsonl',
  trainer+: {
    validation_metric: '+tagging'
  }
}
