(import 'supertagger.jsonnet') + {
  dataset_reader+: {
    type: 'finetune_supertagging_dataset',
    lazy: true,
    // "noisy_weight": 0.4,
    ccgbank_ratio: 0.4,
    auxiliary_ratio: 0.16,
    token_indexers: super.token_indexers,
  },
  train_data_path: {
    ccgbank: 'http://cl.naist.jp/~masashi-y/resources/tagger_data/traindata.json',
    // "tritrain": "http://cl.naist.jp/~masashi-y/resources/tagger_data/headfirst_filtered.json",
    // tritrain: '/home/cl/masashi-y/public_html/resources/tagger_data/4000qs_fixed_and_filtered.quote_fixed.json',
    auxiliary: '/home/cl/masashi-y/public_html/resources/tagger_data/swbd_nxt_train_wo_fail.no_disfl.json'
  },
  validation_data_path: '/home/cl/masashi-y/public_html/resources/tagger_data/swbd_nxt_test.no_disfl.json'
}
