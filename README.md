# KG-BERT: BERT for Knowledge Graph Completion

The repository is modified from [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and tested on Python 3.5+.


## Installing requirement packages

```bash
pip install -r requirements.txt
```

## Data

(1) The benchmark knowledge graph datasets are in ./data. 

(2) entity2text.txt or entity2textlong.txt in each dataset contains entity textual sequences.

(3) relation2text.txt in each dataset contains relation textual sequences.

## Reproducing results
 
### 1. Triple Classification

#### WN11

```shell
python run_bert_triple_classifier.py 
--task_name kg
--do_train  
--do_eval 
--do_predict 
--data_dir ./data/WN11 
--bert_model bert-base-uncased 
--max_seq_length 20 
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 3.0 
--output_dir ./output_WN11/  
--gradient_accumulation_steps 1 
--eval_batch_size 512
```

#### FB13

```shell
python run_bert_triple_classifier.py 
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./data/FB13 
--bert_model bert-base-cased
--max_seq_length 200
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 3.0 
--output_dir ./output_FB13/  
--gradient_accumulation_steps 1 
--eval_batch_size 512
```


### 2. Relation Prediction

#### FB15K

```shell
python3 run_bert_relation_prediction.py 
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./data/FB15K 
--bert_model bert-base-cased
--max_seq_length 25
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 20.0 
--output_dir ./output_FB15K/  
--gradient_accumulation_steps 1 
--eval_batch_size 512
```

### 3. Link Prediction

#### WN18RR

```shell
python3 run_bert_link_prediction.py
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./data/WN18RR
--bert_model bert-base-cased
--max_seq_length 50
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 5.0 
--output_dir ./output_WN18RR/  
--gradient_accumulation_steps 1 
--eval_batch_size 5000
```

#### UMLS

```shell
python3 run_bert_link_prediction.py
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./data/umls
--bert_model bert-base-uncased
--max_seq_length 15
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 5.0 
--output_dir ./output_umls/  
--gradient_accumulation_steps 1 
--eval_batch_size 135
```

#### FB15k-237

```shell
python3 run_bert_link_prediction.py
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./data/FB15k-237
--bert_model bert-base-cased
--max_seq_length 150
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 5.0 
--output_dir ./output_FB15k-237/  
--gradient_accumulation_steps 1 
--eval_batch_size 1500
```
