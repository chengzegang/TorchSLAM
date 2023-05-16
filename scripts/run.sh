SCRIPT_DIR=$(dirname "$0")
source $SCRIPT_DIR/env.sh
cd $SCRIPT_DIR/../src
python -m torchslam wtf $VIDEO_PATH