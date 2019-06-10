(import 'supertagger.jsonnet') + {
  dataset_reader+: {
    type: 'finetune_supertagging_dataset',
    lazy: true,
    // "noisy_weight": 0.4,
    ccgbank_ratio: 0.16,  // for geometry, which contains 63 samples
    auxiliary_ratio: 100,
    token_indexers: super.token_indexers,
  },
  train_data_path: {
    ccgbank: 'http://cl.naist.jp/~masashi-y/resources/tagger_data/traindata.json',
    auxiliary: '/home/cl/masashi-y/public_html/resources/tagger_data/geo-train_fix_np.json',
  },
  validation_data_path: 'http://cl.naist.jp/~masashi-y/resources/tagger_data/devdata.json',
}
