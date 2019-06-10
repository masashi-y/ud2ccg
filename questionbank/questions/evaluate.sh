#!/bin/sh

INPUT_AUTO=$1
INPUT_FILENAME=`basename $INPUT_AUTO`

CANDC=/home/masashi-y/.ghq/github.com/masashi-y/ud2ccg/candc-1.00/candc-1.00/
MARKEDUP=markedup_sd-1.00
export CANDC=$CANDC

sed -e "s/\"/QUOTE/g" $INPUT_AUTO > /tmp/$INPUT_FILENAME
# cp $INPUT_AUTO /tmp/$INPUT_FILENAME
$CANDC/src/scripts/ccg/get_grs_from_auto /tmp/$INPUT_FILENAME $MARKEDUP /tmp/$INPUT_FILENAME.deps
python2 grs2sd-1.00 --ccgbank /tmp/$INPUT_FILENAME.deps > /tmp/$INPUT_FILENAME.sd
# perl -00pe0 's/^\n\n/\n/s' /tmp/$INPUT_FILENAME.sd > /tmp/$INPUT_FILENAME.sd2
python evaluate.py /tmp/$INPUT_FILENAME.sd qus_test.depsGold
