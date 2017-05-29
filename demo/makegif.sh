#! /usr/bin/env bash
set -e

WIDTH=60
HEIGHT=90
SEQLEN=5
SLEEP=150
TOTALWIDTH=`expr ${WIDTH} \* ${SEQLEN}`
TOTALHEIGHT=`expr 2 \* ${HEIGHT} + 10`
OFFSET=100
echo $TOTALHEIGHT
DATA_TYPE=${DATA_TYPE:-reverse}
echo "Using type=${DATA_TYPE}. To change this set DATA_TYPE to 'copy' or 'reverse' or 'add'"

INPUT_DIR=${OUTPUT_DIR:-$HOME/gifs/${DATA_TYPE}}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/gifs/${DATA_TYPE}/outs}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."
mkdir -p $OUTPUT_DIR

convert -delay $SLEEP -loop 0 $INPUT_DIR/itape*.png $OUTPUT_DIR/gifintape.gif
convert -delay $SLEEP -loop 0 $INPUT_DIR/otape*.png $OUTPUT_DIR/gifouttape.gif

convert $OUTPUT_DIR/gifintape.gif -repage ${TOTALWIDTH}x${TOTALHEIGHT} -coalesce null: \( $OUTPUT_DIR/gifouttape.gif \) -geometry +0+$OFFSET -layers Composite $OUTPUT_DIR/${DATA_TYPE}_final.gif



