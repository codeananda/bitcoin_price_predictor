# Let's Predict the Price of Bitcoin with Tensorflow/Keras 📈

> The financial markets generally are unpredictable… The idea that you can actually predict what's going to happen contradicts my way of looking at the market. - George Soros

For this project, I built Tensorflow/Keras models to predict the price of Bitcoin.

## Introduction

One of my mentors had a sequence classification project coming up with a client in a few weeks. He wanted me to demonstrate I could do time series/sequence classification in Tensorflow/Keras. I had never built such a model before, nor had I done a time series problem. Since we both have an interest in cryptocurrency, we thought it would be fun to build a model to predict the price of Bitcoin. 

Disclaimer: since people dedicate their lives to building financial trading models, I thought there was a close to zero percent chance I would build a profitable model. So, I treated this as a project to build my ML skills.

I completed the project much to my mentor's satisfaction (and then we completed the [client's sequence classification project](https://github.com/theadammurphy/sequence_classification_pytorch)) and he left the following ⭐⭐⭐⭐⭐ review:

<img width="636" alt="Screenshot 2021-09-01 at 17 19 46" src="https://user-images.githubusercontent.com/51246969/131698489-32a12020-7cd0-4277-9a62-1907f0eff43d.png">

Note: I spent more than 8 hours on this but asked to bill it hourly to increase the number of hours billed on my [Upwork profile](https://www.upwork.com/freelancers/~01153ca9fd0099730e).

# This Repo is a Work-in-Progress 🏗

I finished this project in June 2021 and am in the process of tidying everything up so it can be presented to the world in a nice manner. You are one of the lucky souls who gets to see the repo in its raw form. But this means that not everything is as clean or orderly as it should be.

# Define

1. Demonstrate I can build and tune Tensorflow/Keras models on time series/sequence classification problems
2. Demonstrate I can work independently each week and report back on progress

# Discover

# Develop

# Deploy

# Results

The best results were obtained by an LSTM with 5 layers each getting sequentially smaller. I ran multiple tests on wandb and got the lowest loss on the validation set to be 0.01816 RMSE.

![X_val predictions for runs 421-423](https://user-images.githubusercontent.com/51246969/131756693-86770d0d-dfac-4060-94ce-45d1bb021e3a.png)
X_val predictions vs. actuals for runs 421-423 (blue = actual, red = predictions)

Due to the stochastic nature of DL models, I ran each experiment at least 10 times. Best results were obtained on runs 413-424 (which you can search for using the regex `41[3-9]|42[0-3]` on the [wandb project page](https://wandb.ai/theadammurphy/bitcoin_price_predictor?workspace=user-theadammurphy)).

The best model was [pretty-vortex-422](https://wandb.ai/theadammurphy/bitcoin_price_predictor/runs/k8h5jmb0).

I manually tuned the learning rate and implemented a [custom learning rate scheduler](https://github.com/theadammurphy/bitcoin_price_predictor/blob/f6e801c4d83d993f53cb2eb70ee748c7499b2ddc/price_predictor/helpers.py#L583-L589). The optimal batch size was 168, the best scaling was to first apply a log transformation, then scale the min/max to (0, 1), finally the Adam optimizer outperformed the others.

![X_train predictions (broken down)_39_1b11b12c](https://user-images.githubusercontent.com/51246969/131756093-60d9c154-f196-4fff-91e7-b27600848301.png)
Pretty-vortex-422 X_train results - actuals vs. predictions

![loss](https://user-images.githubusercontent.com/51246969/131756167-a35bb676-1f40-409a-9271-883ddc6ebcdf.png)
Pretty-vortex-422 Loss

![1-Root_Mean_Squared_Error - Training and Validation_35_d3217c79](https://user-images.githubusercontent.com/51246969/131756200-2b1eab22-fbe3-42f4-8c0e-70c1dd0410b1.png)
Pretty-vortex-422 1-RMSE

<img src="https://user-images.githubusercontent.com/51246969/131756215-e955cdde-1a67-4956-9cfa-87655b46002c.png" alt="Pretty-vortex-422 X_val results" width=600 />
Pretty-vortex-422 X_val results - actuals vs. predictions - note the low RMSE of 0.01816 and how the red line tightly hugs the blue.


# Notes

### Where is the Code?

The vast majority of the code I used is in [`price_predictor/helpers.py`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/main/price_predictor/helpers.py). I split the functions up into sections and wrote a [`train_and_validate`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/50f726064d2230d748309420716758983909bba0/price_predictor/helpers.py#L895-L940) function to perform all the training and validation steps I wanted to do for each experiment. The functions that make up [`train_and_validate`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/50f726064d2230d748309420716758983909bba0/price_predictor/helpers.py#L895-L940) should make it clear what is happening at each step. 

### Model Tuning

All model tuning for this project took place with [Weights & Biases](https://wandb.ai/site) (wandb). You can see the results of all 540+ runs on the [bitcoin_price_predictor wandb page](https://wandb.ai/theadammurphy/bitcoin_price_predictor?workspace=user-theadammurphy). As such, the notebooks themselves are not that interesting - I just used them to run wandb experiments and saved everything to the cloud. 


# Improvements

### Using Classes

This was my first time building such a model with TensorFlow/Keras. Since then I have used PyTorch Lightning and love the flexibility of their [Data Modules](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html) to encapsulate all the data processing code. I would like to encapsulate more of the code into easy-to-transport classes instead of the (rather large) collection of functions I wrote. 

### Regular Re-Training and Deployment. 

Since the Bitcoin price never stops, it's easy to re-train the model and see how it performs on brand new data. Because we only used the price of Bitcoin to make predictions, I doubt the model will perform well. But it would be great to get a measure of how it performs in production. 

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
