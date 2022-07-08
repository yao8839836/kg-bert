export CUDA_VISIBLE_DEVICES=3

python3 run_bert_link_prediction.py --task_name kg  --do_train  --do_eval --do_predict --data_dir ./data/FB15k-237--bert_model bert-base-cased--max_seq_length 150--train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 5.0 --output_dir ./output_FB15k-237/  --gradient_accumulation_steps 1 --eval_batch_size 1500