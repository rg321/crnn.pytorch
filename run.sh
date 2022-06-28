

python train.py \
--workers 1 --displayInterval 1 \
--alphabet "0123456789abcdefghijklmnopqrstuvwxyz.-'" \
--trainroot data/train$1 --valroot data/test$1 \
--valInterval 10