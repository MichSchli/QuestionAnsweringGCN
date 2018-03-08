#!/bin/bash

TO_BE_RECASED=$1

MOSES_DIR='/home/michael/Projects/Moses/mosesdecoder'
TEMP_FILE=$TO_BE_RECASED'.flat'
UPPERCASED_TEMP_FILE=$TEMP_FILE'.uppercased'
OUTPUT=$TO_BE_RECASED'.uppercased'

python3 helpers/flatten_conll.py --input_file $TO_BE_RECASED > ${TEMP_FILE}

perl $MOSES_DIR/scripts/recaser/recase.perl --in $TEMP_FILE --model /home/michael/Data/MT/recaser.2008-news.model/moses.ini --moses $MOSES_DIR/bin/moses2 > ${UPPERCASED_TEMP_FILE}

python3 helpers/unflatten_conll.py --structure $TO_BE_RECASED --flat $UPPERCASED_TEMP_FILE > ${OUTPUT}

rm $TEMP_FILE
rm $UPPERCASED_TEMP_FILE