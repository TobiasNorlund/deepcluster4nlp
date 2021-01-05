# deepclustering4nlp


# run deepcluster training
python3 main.py --exp "/opt/models/deepcluster/exp" --arch "textcnn" --lr 0.05 --wd -5 --k 100 --verbose --workers 12


# train downstream model
python3 main.py --model_path PATH --exp /opt/models/deepcluster/exp

`model_path` needs to point to a pretrained model

`exp` is where the checkpoints of the downstream model will be saves

You will need to change the code in main.py so that `train_supervised_model()` is run when running the script


There are unfortunately quite a lot of hardcoded elements to this code, so one needs to be patient while making this work.
Some hardcoded parameters for example need to be the same when pretraining as when training the downstream model.

___

# Progress & Conclusions

In the project, we started from the code of the original paper. 
We then modified this code to work on text instead of images.
Our use case is currently classifying sentiment on the IMDB dataset.

We were able to pretrain a bunch of models and store them.
Then we picked a model, loaded it, and started finetuning it to the downstream task of IMDB classification.

Unfortunately, the results we ended up with so far are not useful. 
The pretrained models' weights exploded while training, which made it impossible for us to use that to fit to the 
downstream task.

If one would continue with this project, we would need to debug the pretraining, and make sure that the 
weights do not diverge so much.

The way the code is written makes it very hard to do that.
We believe it would be a good idea to rewrite it from scratch. 
This is of course a lot of effort, with possibly no amazing results, but then you could at least continue 
experimenting.




