# mstcn
python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAE.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAE.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPC.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPC.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/MSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAE.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAE.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPC.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPC.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml



