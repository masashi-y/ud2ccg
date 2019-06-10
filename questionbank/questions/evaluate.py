
from typing import NamedTuple
import argparse

class Token(NamedTuple):
    word: str
    pos: str
    cat: str


def read_deps(filename, skip=None):
    lines = [line.strip() for line in open(filename)]
    if skip:
        lines = lines[skip:]

    res = []
    tmp = set()
    for line in lines:
        if len(line) == 0 or line.startswith('<!'):
            pass
        elif line.startswith('<c>'):
            tokens = [Token(*token.split('|')) for token in line.split(' ')[1:]]
            res.append((tokens, tmp))
            tmp = set()
        else:
            assert line.startswith('(') and line.endswith(')')
            items = tuple(line[1:-1].split(' '))
            if len(items) == 3:
                tmp.add(items)
    return res


parser = argparse.ArgumentParser('question parsing evaluation')
parser.add_argument('PRED_SD')
parser.add_argument('GOLD')
parser.add_argument('--skip', default=4)
args = parser.parse_args()

preds = read_deps(args.PRED_SD, skip=args.skip)
golds = read_deps(args.GOLD)
assert len(preds) == len(golds)

unlabeled_correct = 0
unlabeled_incorrect = 0
unlabeled_missing = 0

labeled_correct = 0
labeled_incorrect = 0
labeled_missing = 0

cat_correct = 0
cat_all = 0
skipped = []
for i, ((tokens_pred, pred), (tokens_gold, gold)) in enumerate(zip(preds, golds)):
    cats_pred = [token.cat for token in tokens_pred]
    cats_gold = [token.cat for token in tokens_gold]
    assert len(cats_pred) == len(cats_gold)
    cat_correct += sum(1 if pred == gold else 0 for pred, gold in zip(cats_pred, cats_gold))
    cat_all += len(cats_gold)

    if len(pred) == 0:
        print(tokens_pred)
        skipped.append(i)
        continue
    unlabeled_pred = {(start, end) for _, start, end in {rel for rel in pred if len(rel) == 3}}
    unlabeled_gold = {(start, end) for _, start, end in {rel for rel in gold if len(rel) == 3}}
    unlabeled_correct += len(unlabeled_gold.intersection(unlabeled_pred))
    unlabeled_incorrect += len(unlabeled_pred.difference(unlabeled_gold))
    unlabeled_missing += len(unlabeled_gold.difference(unlabeled_pred))

    correct = gold.intersection(pred)
    in_correct = pred.difference(gold)
    missing = gold.difference(pred)

    labeled_correct += len(correct)
    labeled_incorrect += len(in_correct)
    labeled_missing += len(missing)

    if len(in_correct) > 0 or len(missing) > 0:
        print(i, ':', ' '.join(token.word for token in tokens_pred))
        # print('correct:', correct)
        print('precision error:', in_correct)
        print('recall error:', missing)
        print()


unlabeled_precision = float(unlabeled_correct) / float(unlabeled_correct + unlabeled_incorrect)
unlabeled_recall = float(unlabeled_correct) / float(unlabeled_correct + unlabeled_missing)
unlabeled_f1 = (2 * unlabeled_precision * unlabeled_recall) / (unlabeled_precision + unlabeled_recall)

labeled_precision = float(labeled_correct) / float(labeled_correct + labeled_incorrect)
labeled_recall = float(labeled_correct) / float(labeled_correct + labeled_missing)
labeled_f1 = (2 * labeled_precision * labeled_recall) / (labeled_precision + labeled_recall)
print(f'precision: {unlabeled_precision}')
print(f'recall: {unlabeled_recall}')
print(f'f1: {unlabeled_f1}')

print(f'skipped: {skipped}')
print(f'precision: {labeled_precision}')
print(f'recall: {labeled_recall}')
print(f'f1: {labeled_f1}')


cat_accuracy = float(cat_correct) / float(cat_all)
print(f'accuracy: {cat_accuracy}')
