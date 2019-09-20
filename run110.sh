#!/bin/bash

#is_test=1: lancia un run di prova
#is_test=0: run normale
#fold=10: numero di fold per la cross validation (se cambiato si deve cambiare anche i file 10fold_idx)
#repeted: numero di ripetizioni della cross validation
#bsize: grandezza del batch per il learning
#p1 e p2: dimensioni laterali della SOM
#sigma_out: sigma della funzione output per la SOM
#wd: weight decay
#reg: parametro per la regolarizzazione L2 dell'ultimo layer

is_test=0
fold=10
repeted=3
learning_rate=0.0001
num_epochs=800
if [ $is_test -eq 1 ]
then
    num_epochs=1
fi
bsize=56
p1=6
p2=8
sigma_out=1
wd=0.000001
reg=0.000005

python accuracy_early_stopping.py \
    -is_test $is_test \
    -fold $fold \
    -repetitions $repeted \
    -learning_rate $learning_rate \
    -num_epochs $num_epochs \
    -batch_size $bsize \
    -p1 $p1 \
    -p2 $p2 \
    -sigma_out $sigma_out \
    -weight_decay $wd \
    -regularization $reg


