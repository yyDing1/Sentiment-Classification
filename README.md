# Sentiment-Classification


## Prepare the Environment

```bash
bash install_tool.sh
```

## Run

You can run the whole training and test stage by running

```bash
python LSTM-solution.py --do_train --do_test [--bidirectional]
```


if you already prepare the Word2Vec Model, you can load the pretrained word embedding by running

```bash
python LSTM-solution.py --do_train --w2vmodel_path $W2VMODEL_PATH --do_test [--bidirectional]
```

you may only train or test by running

```bash
python LSTM-solution.py [--do_train] [--do_test] [--bidirectional] --model_save_path $MODEL_SAVE_PATH

```

BERT-base-Chinese: 

```bash
python BERT-solution.py --do_train --do_test --model_name_or_path bert-base-chinese
```

BERT--wwm

```bash
python BERT-solution.py --do_train --do_test --model_name_or_path bert-wwm
```

Detailed hype-parameters settings can be seen in `LSTM-solution.py` and `BERT-solution.py`

