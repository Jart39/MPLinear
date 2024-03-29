export CUDA_VISIBLE_DEVICES=0

model_name=MPLinear

python -u run_new.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --model_id weather_96_96 \
 --model $model_name \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 96 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 1 \
 --train_epochs 8 \
 --d_model 260 \
 --top_k 5 \
 --dropout 0.1 \
 --dropout1 0.1 \
 --lradj 'FFT' \
 --batch_size 64
