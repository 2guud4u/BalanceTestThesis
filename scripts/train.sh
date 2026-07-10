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
# bilstm
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

# asformer

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAE.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAE.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPC.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/GCN.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPC.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml



# better mstcn config

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/smallerMSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MB.yml \
    --encoder /code/jjiang23/BalanceTestThesis/configs/encoder/MAMP.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/smallerMSTCN.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/upsample.yml 

# bottom half vs top half

python3 train.py \
    --data /code/jjiang23/BalanceTestThesis/configs/data/MPWBottomHalf.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/BalanceTestThesis/configs/data/MPWTopHalf.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/ASFormer.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml

python3 train.py \
    --data /code/jjiang23/BalanceTestThesis/configs/data/MPWBottomHalf.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml && \

python3 train.py \
    --data /code/jjiang23/BalanceTestThesis/configs/data/MPWTopHalf.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/biLSTM.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/downsamp.yml

# hyper

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/asf_hyper.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/asf_hyper.yml \
    --exclude_folds fold_0

python3 train.py \
    --data /code/jjiang23/pathml/aim2_balanceV2/configs/data/MPW.yml \
    --segmentor /code/jjiang23/BalanceTestThesis/configs/segmentor/bilstm_hyper.yml \
    --trainer /code/jjiang23/BalanceTestThesis/configs/trainer/bilstm_hyper.yml \
    --exclude_folds fold_0




