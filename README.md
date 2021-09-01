# Let's Predict the Price of Bitcoin 📈
For this project, I build models in Tensorflow/Keras to predict the price of Bitcoin over the next few days. I used fully-connected neural nets and LSTMs.

This was a paid project on Upwork that lasted several months. The client wanted a proof-of-concept model, so we stuck to univariate time-series forcasting. We were both happy with the results and the client left the following ⭐⭐⭐⭐⭐ review:

<img width="636" alt="Screenshot 2021-09-01 at 17 19 46" src="https://user-images.githubusercontent.com/51246969/131698489-32a12020-7cd0-4277-9a62-1907f0eff43d.png">

Note: we decided beforehand it would be a fixed price project but I asked to bill it hourly to increase the number of hours billed on my [Upwork profile](https://www.upwork.com/freelancers/~01153ca9fd0099730e).

# This Repo is a Work-in-Progress 🏗

I finished this project in June 2021 and am in the process of tidying everything up so it can be presented to the world in a nice manner. You are one of the lucky souls who gets to see the repo in its raw form. But this means that not everything is as clean or orderly as it should be.

However, I hope it gives you an idea of how I approached this project and demonstrates I can build and tune Tensorflow/Keras models on univariate time-series data.

# Results

The best results were obtained by an LSTM with 5 layers each getting sequentially smaller. I ran multiple tests on wandb and achieved a low loss on the validation set of

<img width="1252" alt="Screenshot 2021-09-02 at 00 28 27" src="https://user-images.githubusercontent.com/51246969/131754518-df0674f4-4f28-49ec-860e-fb66ee2a75ef.png">

# Notes

### Where is the Code?

The vast majority of the code I used is in [`price_predictor/helpers.py`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/main/price_predictor/helpers.py). I split the functions up into sections and wrote a [`train_and_validate`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/50f726064d2230d748309420716758983909bba0/price_predictor/helpers.py#L895-L940) function to perform all the training and validation steps I wanted to do for each experiment. The functions that make up [`train_and_validate`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/50f726064d2230d748309420716758983909bba0/price_predictor/helpers.py#L895-L940) should make it clear what is happening at each step. 

### Model Tuning

All model tuning for this project took place with [Weights & Biases](https://wandb.ai/site) (wandb). You can see the results of all 540+ runs on the [bitcoin_price_predictor wandb page](https://wandb.ai/theadammurphy/bitcoin_price_predictor?workspace=user-theadammurphy). As such, the notebooks themselves are not that interesting - I just used them to run wandb experiments and saved everything to the cloud. 

### Docstrings

Not all functions have docstrings and for this I will probably face 🔥 eternal damnation in hell 🔥. In my defence, I was working on this project, alone every day and felt like the names were clear enough. 

🙏 **I promise to add docstrings when I write everything up neatly.** 🙏

In the meantime, check out the scripts from my [PyTorch project](https://github.com/theadammurphy/portfolio/tree/main/electrochem_pytorch/scripts) if you want proof I can write docstrings.

# Improvements

### Using Classes

This was my first time building such a model with TensorFlow/Keras. Since then I have used PyTorch Lightning and love the flexibility of their [Data Modules](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html) to encapsulate all data processing code. I would like to encapsulate more of the code into easy-to-transport classes rather than the rather the large collection of functions I wrote. 

### Regular Re-Training and Deployment. 

Since the Bitcoin price never stops, it's easy to re-train the model and see how it performs on brand new data. Because we only used the price of Bitcoin to make predictions, I doubt the model will perform well. But it would be great to get a measure of how well (or poorly) it performs in production. 

## Libraries Used

I used Python and the following libraries:
* TensorFlow (and Keras) 2.4 
* Numpy
* Pandas
* Scikit-learn 
* Wandb
* Matplotlib
* Seaborn
* Tqdm
