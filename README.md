# Let's Predict the Price of Bitcoin 📈
For this project, I build a model in Tensorflow/Keras to predict the price of Bitcoin over the next few days. I used Deep Neural Nets and LSTMs.

This was a paid project on Upwork that lasted several months. The client wanted a proof-of-concept model, so we just stuck to univariate time-series forcasting. We were both very happy with the results and the client left the following ⭐⭐⭐⭐⭐ review:

<img width="636" alt="Screenshot 2021-09-01 at 17 19 46" src="https://user-images.githubusercontent.com/51246969/131698489-32a12020-7cd0-4277-9a62-1907f0eff43d.png">

# Notes

All model tuning for this project took place with [Weights & Biases](https://wandb.ai/site) (wandb). You can see the results of all 540+ runs on the [bitcoin_price_predictor wandb page](https://wandb.ai/theadammurphy/bitcoin_price_predictor?workspace=user-theadammurphy). 

As such, the notebooks themselves are not that interesting (since I just used them to run wanbd experiments and everything was saved to the cloud). 

The vast majority of the code I used is in `price_predictor/helpers.py`. I split the functions up into sections and wrote a [`train_and_validate`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/50f726064d2230d748309420716758983909bba0/price_predictor/helpers.py#L895-L940) function to perform all the training and validation steps I wanted to do for each experiment. The function names that make up [`train_and_validate`](https://github.com/theadammurphy/bitcoin_price_predictor/blob/50f726064d2230d748309420716758983909bba0/price_predictor/helpers.py#L895-L940) should make ti clear what is happening at each step. 

I know that not all functions have docstrings yet and for this I will probably face eternal damnation in hell. But, I was working on this project every day and felt like the names were clear enough. I will add docstrings when I write everything up neatly, I promise! In the meantime, check out the scripts from my [PyTorch project](https://github.com/theadammurphy/portfolio/tree/main/electrochem_pytorch/scripts) for proof that I can write docstrings.

This repo has not been edited or cleaned. In the near future, I will turn this into a well-written portfolio project. But, for now, here it is in its raw form.

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
