# kaggle_OTTO
## Task and solution overview.
This is the code for kaggle competition called "OTTO – Multi-Objective Recommender System". OTTO is Germany's largest online retailer. The task is to predict, which exact next item user is going to click next and which items user is going to add to cart or order before the end of the test period. The competition data is real life new users' sessions on OTTO website. The test dataset includes one week of users' sessions, truncated at a random point. Organizers also provide participants with history of full user's sessions for four weeks, preceding the test period. But no metainformation is available for all the items, that show up in both datasets. Participants only have item ID's, that are called AIDs, and there are about 1.8 mln AIDs showing up in the competition data, including both full and truncated user sessions.

From the beginning I have decided to use Jupyter notebooks ran on kaggle website to produce the solution. However, this decision turned out to have a number of complications. Firstly, kaggle notebooks without GPU support at a time of competition had RAM limit of 30 Gb and I had to spend some time, for example, writing code that would merge two dataframes chunk by chunk, as using a simple merge would cause a memory error. Kaggle notebooks with GPU available have RAM limit of just 13 Gb, that made me choose between some features, instead of using all of them. Then, Jupyter notebooks are not that useful when dealing with projects that require a complicated data pipeline. Total number of notebooks used to produce the final solution is 28, and that number does not include a few more notebooks used to test some approaches that brought no fruit.

The solution's pipeline includes following major stages:
* creating a cross-validation dataset from the last week of known full sessions, similar to the test dataset;
* calculating co-visitation matrixes, word2vec models and making some other calculations aside of the main pipeline;
* generating candidates;
* engineering features;
* training the GBDT re-ranking models on the cross-validation dataset and using those models to select most relevant candidates, generated for the test sessions;
* some final formatting and submitting the results.
Generating candidates, engineering features, training models and making final prediction stages are separate for each prediction, i.e. for clicks, carts and orders. For each of these predictions, a dedicated model is being trained on a different dataset and for different candidates, although many features are actually common.

## A closer look at input data and metric.
Each session data in the inputs consists of a session ID and sequence of events, each event includes AID, a timestamp, and event type, which could be either click, cart or order. There are total 12.9 mln full sessions, and a median full session has 6 events, while there are 1.7 mln truncated sessions in the test dataset and a median truncated session has just 2 events. More than 90% of all events are clicks, most sessions are short and only include clicks. 

For every session in the test dataset, competitors predict 20 clicks, 20 carts and 20 orders. Here is the formula used to score the predictions:

score = 0.1*R<sub>clicks</sub> + 0.3*R<sub>carts</sub> + 0.6*R<sub>orders</sub>,

and each of the R values is a recall that could take values between 0 and 1. So, the coefficients are set in a way that makes predicting orders more important than predicting carts, and predicting carts more important than predicting clicks.

## All project notebooks
* notebooks with common code
  * OTTO common
  * OTTO common feature engineering
* creating a cross-validation dataset
  * Prepare cross-validation (otto-prepare-cv.ipynb)
* calculations aside of the main pipeline
  * "Regular" co-visitation click2click matrix (otto_click2click_regular.ipynb)
  * "Experimental" co-visitation click2click matrix (otto-click2click-experiment.ipynb)
  * Click2buy and buy2buy co-visitation matrixes (otto-click2buy-buy2buy.ipynb)
  * Click2buy short co-visitation matrix (otto-click2buy-short.ipynb)
  * W2vec model for clicks (otto-word2vec-clicks.ipynb)
  * W2vec model for carts and orders (otto-word2vec-carts-orders.ipynb)
  * Calculations for clicks (create-counts-for-clicks.ipynb)
  * Calculations for buys (create-counts-buys.ipynb)
* generating candidates
  * Generate candidates for clicks
  * Generate candidates for carts
  * Generate candidates for orders
* engineering features
  * Feature engineering for clicks model
  * Feature engineering for carts model
  * Feature engineering for orders model
  * W2vec features for clicks
  * W2vec features for carts
  * W2vec features for carts (part_1)
  * W2vec features for orders
  * W2vec features for orders (part_1)
* training and predicting
  * Clicks Model and Prediction
  * Carts Model
  * Orders Model
  * Carts Prediction
  * Making and combining predictions for orders
* final formatting and submitting the results
  * OTTO Upload

## Cross-validation datasets
Organizers have published the code they have used to produce the test dataset. It actually cuts all the sessions that started before the test period and continue into test period. Then, it selects sessions that have started during test period and filters out sessions with aids not met in any session before test period (short sessions, of course, are more likely to pass through that filter). After that unfiltered sessions are truncated at a random point, leaving at least one known and at least one unknown event. As an output we have a shortened file of full sessions, a cross-validation file of truncated sessions and a file with labels.

I used this code to produce 2 different cross-validation datasets with different random seeds. The intension was to check at some point wether results for different cross-validation sets differ and probably to try using features, generated for two datasets to train two models and then take average prediction. When working on the project I have compared several times intermediate results for the two datasets, and there never was a significant difference. I also tried using two models build upon different cross-validation datasets to predict orders, but improved results just a little bit, so working with two cross-validation dataset probably was not worth the effort.

In the same notebook I also converted all the data, including inputs and cross-validation sets, from json to parquet, changed datatypes from int64 to int32, and mapped event types to integers (0 - clicks, 1 - carts, 2 - orders) to reduce memory usage.

Now with cross-validation datasets ready it is possible to get additional insights on the test dataset and on the answers I am going to predict.
Number of full sessions in history has reduced to 10.6 mln, and the cross-validation datasets have 1.8 mln truncated sessions. 
We can see that for the vast majority of sessions do not have any labels neither for carts nor for orders, while almost all sessions have a click label. This means, cart and order predictions for most short sessions does not matter, as only predictions for sessions with some actual carts or orders give points. Almost all the sessions do have a single ground truth value for clicks, while very few have no ground truth values.

## Calculations aside of the main pipeline
These notebooks include notebooks calculating the co-visitaion matrixes, w2vec models and all the other calculations combined into 2 notebooks "create counts for clicks" and "create counts for buys" (buys means any non-click event, i.e. either cart or click). All the calculations in those notebooks are repeated at least twice: once for the cross-validation dataset and then for the full data, in a few cases the calculations need to be made separately for each cross-validation dataset.

Code to calculate the co-visitation matrix is 90% the same for all the matrixes I have tried for this project. So, I've wrote a class and moved it to the OTTO common notebook, so that it could be used to create a child class for each case to count different type of co-occuring events. These notebooks have very little code and they take 2 to 4 hours to run.

Here is the full list of co-visitation matrixes, used to produce the final result:
* **"Regular" co-visitation click2click matrix** (otto_click2click_regular.ipynb) counts events of any type in a session, if time difference between them is less than 5 minutes. Weight coefficient is calculates in a way that makes later events to have higher weights. Here, the ordering of events does not count for this matrix, this means that if in a session  This matrix is used both for click candidates generation and to calculate a feature (wgt_matrix) for the clicks model.
* **"Experimental" co-visitation click2click matrix** (otto-click2click-experiment.ipynb) counts events of any type in a session 
