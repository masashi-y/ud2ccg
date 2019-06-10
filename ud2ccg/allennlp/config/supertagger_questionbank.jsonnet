(import 'supertagger.jsonnet') + {
  dataset_reader+: {
    type: 'finetune_supertagging_dataset',
    lazy: true,
    // "noisy_weight": 0.4,
    ccgbank_ratio: 0.25,  // questionbank contains 3622 training examples
    tritrain_ratio: 2,
    auxiliary_ratio: 1,
    token_indexers: super.token_indexers,
  },
  train_data_path: {
    ccgbank: 'http://cl.naist.jp/~masashi-y/resources/tagger_data/traindata.json',
    tritrain: '/home/cl/masashi-y/ud2ccg/questions/questions/qus_train.json',
    auxiliary: '/home/cl/masashi-y/public_html/resources/tagger_data/4000qs_fixed_and_filtered.quote_fixed.json',
  },
  validation_data_path: 'http://cl.naist.jp/~masashi-y/resources/tagger_data/devdata.json',
}
