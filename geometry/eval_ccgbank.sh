EASYCCG=$HOME/easyccg/
CANDC=$HOME/candc/candc-1.00/
GOLDSTAGGED=$EASYCCG/working/gold/wsj23.stagged
GOLDDEPS=/home/cl/masashi-y/ud2ccg/geo/geo-dev.gold_deps

INPUT=$1
DEPS=$1.deps

export CANDC

$EASYCCG/training/eval_scripts/get_deps_from_auto $INPUT $DEPS

sed -i -e 's/^$/<c>\n/' $DEPS

$EASYCCG/training/eval_scripts/evaluate3 -r $GOLDSTAGGED $GOLDDEPS $DEPS
