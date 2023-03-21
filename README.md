# kaggle_OTTO
## Task and solution overview.
This is the code for kaggle competition called "OTTO – Multi-Objective Recommender System". OTTO is Germany's largest online retailer. The task is to predict which exact next item user is going to click next and which items user is going to add to cart or order before the end of the test period. The competition data is real life new users' sessions on OTTO website. The test dataset includes one week of users' sessions, truncated at a random point. Organizers also provide participants with history of full user's sessions for four weeks, preceding the test period. But no metadata is available for all the items, that show up in both datasets. Participants only have item ID's, that are called AIDs, and there are about 1.8 mln AIDs showing up in the competition data, including both full and truncated user sessions.

From the beginning I have decided to use Jupyter notebooks ran on kaggle website to produce the solution. However, this decision turned out to have a number of complications. Firstly, kaggle notebooks without GPU support at a time of competition had RAM limit of 30 Gb and I had to spend some time, for example, writing code that would merge two dataframes chunk by chunk, as using a simple merge would cause a memory error. Kaggle notebooks with GPU available have RAM limit of just 13 Gb, that made me choose between some features, instead of using all of them. Then, Jupyter notebooks are not that useful when dealing with projects that require a complicated data pipeline. Total number of notebooks used to produce the final solution is 28, and that number does not include a few more notebooks used to test some approaches that brought no fruit.

The solution's pipeline includes the following major stages:
* creating a cross-validation dataset from the last week of known full sessions, similar to the test dataset;
* calculating co-visitation matrixes, word2vec models and making some other calculations aside of the main pipeline;
* generating candidates;
* engineering features;
* training the GBDT re-ranking models on the cross-validation dataset and using those models to select most relevant candidates, generated for the test sessions;
* some final formatting and submitting the results.
Generating candidates, engineering features, training models and making final prediction stages have separate notebooks for clicks, carts and orders. For each of these predictions, a dedicated model is being trained, with different features, on a different dataset and for different candidates, although some features are actually common.

## A closer look at input data and metric.
Each session data in the inputs consists of a session ID and sequence of events, each event includes AID, a timestamp, and event type, which could be either click, cart or order. There are total 12.9 mln full sessions, and a median full session has 6 events, while there are 1.7 mln truncated sessions in the test dataset and a median truncated session has just 2 events. More than 90% of all events are clicks, most sessions are short and only include clicks. 

For every session in the test dataset, competitors predict 20 clicks, 20 carts and 20 orders. Here is the formula used to score the predictions:

score = 0.1*R<sub>clicks</sub> + 0.3*R<sub>carts</sub> + 0.6*R<sub>orders</sub> , 

and each of the R values is a recall that could take values between 0 and 1. So, the coefficients are set in a way that makes predicting orders more important than predicting carts, and predicting carts more important than predicting clicks.

## All project notebooks
Here is the full list of notebooks, used in the project pipeline:
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
  * Generate candidates for clicks (otto-click-candidates-generation.ipynb)
  * Generate candidates for carts (otto-generate-candidates-carts.ipynb)
  * Generate candidates for orders (otto-generate-candidates-orders.ipynb)
* engineering features
  * Feature engineering for clicks model (otto-feature-engineering-clicks.ipynb)
  * Feature engineering for carts model (otto-feature-engineering-carts.ipynb)
  * Feature engineering for orders model (otto-feature-engineering-orders.ipynb)
  * W2vec features for clicks (otto-clicks-w2vec.ipynb)
  * W2vec features for carts (otto-carts-w2vec.ipynb)
  * W2vec features for carts (part_1) (otto-carts-w2vec-part1.ipynb)
  * W2vec features for orders (otto-orders-w2vec.ipynb)
  * W2vec features for orders (part_1) (otto-orders-w2vec-part1.ipynb)
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
Out of that number only about 300,000 sessions or 17% have at least one aid carted and about half of than number, about 150,000 sessions or 8% have at least one aid ordered. The vast majority of sessions do not have any labels neither for carts nor for orders. This means, cart and order predictions for most short sessions does not matter, as only predictions for sessions with some actual carts or orders give points. At the same time, almost all the sessions do have a single ground truth value for clicks, while very few have no ground truth values.

## Calculations aside of the main pipeline
These notebooks include notebooks calculating the co-visitaion matrixes, w2vec models and all the other calculations combined into 2 notebooks "create counts for clicks" and "create counts for buys" (buys means any non-click event, i.e. either cart or click). All the calculations in those notebooks are repeated at least twice: once for the cross-validation dataset and then for the full data, in a few cases the calculations need to be made separately for each cross-validation dataset.

Code to calculate the co-visitation matrix is 90% the same for all the matrixes I have tried for this project. So, I've wrote a class and moved it to the OTTO common notebook, so that it could be used to create a child class for each case to count different type of co-occuring events. These notebooks have very little code and they take 2 to 4 hours to run.

Here is the full list of co-visitation matrixes, used to produce the final result:
* **"Regular" co-visitation click2click matrix** (otto_click2click_regular.ipynb) counts two events of any type in a session, if time difference between them is less than 5 minutes. Weight coefficient is calculated in a way that makes later events to have higher weights. The ordering of events does not count for this matrix, this means that if two events are close to each other in the same session, it does not matter which one of them comes first. This matrix is used both for click candidates generation and to calculate a feature (wgt_matrix) for the clicks model.
* **"Experimental" co-visitation click2click matrix** (otto-click2click-experiment.ipynb) counts two events of any type in a session, if time difference between them is less than 5 minutes and there are no more than 20 events between them. Weight coefficient is calculated in a way that makes later events to have higher weights. The ordering of events does count for this matrix, this means that aid_y is only counted if it comes after aid_x. I tried to use this matrix to generate click candidates, but "regular" matrix showed better result. In the final pipeline this matrix is used to calculate a feature (wgt_exp) for the clicks model.
* **Click2buy co-visitation matrix** (otto-click2buy-buy2buy.ipynb) counts events in a session, if the later event is a buy (either cart or order) and time difference between them is less than 10 hours. The weight value is calculated in a way that makes pairs of events with smaller time difference more important. This matrix is used both for carts and orders candidate generation and to calculate 2 features (wgt_c2buy_full and wgt_c2buy_6_from_full) for carts and orders models.
* **Buy2buy co-visitation matrix** (otto-click2buy-buy2buy.ipynb) counts events in a session, if both of them are buys and time difference between them is less than 5 days. The weight value is always equal to, so each pair of events is equally important. This matrix is used to calculate a feature (wgt_buy2buy) for carts and orders models.
* **Click2buy short co-visitation matrix** (otto-click2buy-short.ipynb) counts events in a session, if the later event is a buy (either cart or order) and time difference between them is less than 2 hours. The weight value is calculated in a way that makes pairs of events with smaller time difference more important. Comparing to previously mentioned click2buy co-visitation matrix, in this one time difference is limited to much shorter time and the weight value declines much faster as time between the events increases. This matrix is used to build a feature (wgt_c2buy_short) for carts and orders models.
* **Exact next click-to-click co-visitation matrix** (create-counts-for-clicks.ipynb) counts only exact next, regardless of their type or of time passed between them. This is the fastest matrix to calculate, so it doesn't have a dedicated notebook, and is calculated in the same notebook with some other side calculations for the clicks model. This matrix is used to calculate two features for the clicks model: 'wgt_last' and 'wgt_before_last'.

Notebooks with w2vec model generation both are very short in terms of number of rows with code. Information about event type and event time is removed, so, the sequence of aids is the only information kept. That information is passed to a standard function that trains the w2vec model. But these notebooks take significant time to run: it takes about 3 hours to train models for cross-validation and test datasets, using first 3 weeks of full sessions or all the known full sessions correspondingly.

I had an intuition, that a w2vec model with a longer window would be more usefull for carts and orders models, while model with shorter window would produce better results for the clicks model. I've made the checks during the competition, trying both w2vec models to produce features for each of GBDT models and confirmed that this is true. So, I kept using two different w2vec models trained with slightly different parameters. However, difference in performance between the two w2vec models was relatively small, so I choose not to make any additional experiments with changing the models' parameters and tried some other ideas instead.

List of side calculations made in "Calculations for buys" notebook:
* conversion rate - means conversion from click to either cart or order;
* conversion to carts - conversion from either clicks, previously carted aids or previously ordered aids to carts;
* conversion to orders - conversion from either clicks, carts or previously ordered aids to new orders;
* average per aid clicks before buy;
* daily total carts/orders per aid;
* average w2vec similarity between the last one aid in session and 5 aids before it.

List of side calculations made in "Calculations for clicks" notebook:
* median time users view aid;
* average per day clicks per aid;
* return rate, counting how often users return for a new click or other actions with the same aid;
* exact next click-to-click co-visitation matrix, that has been already mentioned earlier.

## Generating candidates
The generation of candidates is rules-based for the clicks, carts and orders. I've spent significant time trying to improve candidate generation process, probably put too much effort in it. For all the candidate generations I generally use three sources of candidates: session history aids, co-visitaion matrixes and daily most popular aids. Depending on number of candidates I use different hand-picked coefficients that define how many candidates come from each source. I've started with using 50 candidates for all the three models, then moved to 75 candidates both for carts and orders. I've planned to start using 75 candidates also for the clicks model. But clicks model has the lowest coefficients in the competition metric, but at the same time it is the most demanding model in terms of memory usage. So I kept using only 50 candidates for the clicks model.

For clicks model I use the lowest number of aids from session history, as the model is aimed at guessing the exact next aid clicked, so aids clicked some time ago are  usually less relevant. So, in case of generating candidates for clicks I take latest aids from session history, then add aids suggested in the co-visitation matrix for the exact last aid, then add to the list most common aids suggested by the co-visitation matrix for a few last aids in session history. Then I remove duplicates from the list and cut it to get the desired number of candidates. If after removing all the duplicates there are less aids in the list then the desired number of candidates, then I one by one add aids from daily top of most popular aids (after checking for each one that it not in the list already). For 20 click candidates my best result was 52.68% percent guessed, while for 50 candidates it was 60.43%.

For cart and order candidates I also take latest aids from session history, first latest buys, then all the latest aids, then add most common aids suggested in the co-visitation matrix for last buys, then add to the list most common aids suggested by the co-visitation matrix for any last aid in the session history. All the constants, like number of buys to take from session history, number of aids from session history, maximum number of aids, suggested from buys e.t.c. vary for carts/orders and depending on number of candidates, but the logic is mostly the same. Then, like when generating candidates for clicks, I remove duplicates from the list and cut it to get the desired number of candidates. If after removing all the duplicates there are less aids in the list then the desired number of candidates, then I one by one add aids from daily top of most popular aids (after checking for each one that it not in the list already). For 20 cart candidates my best result was 40.68% percent guessed, while for 75 candidates it was 47.12%. For orders, best result for 20 candidates was 64.84%, while for 75 candidates it was 68.95%.

We can see, that percent of guessed orders is much higher, than percent of guessed carts. This is mostly because all the carted aids have a very high chance to be ordered, i.e. you make an obvious move - suggest all previously carted aids are going to be ordered, and get a good percent of guessed items. But it is harder to guess carts, as recently viewed aids have a much lower chance to be added to cart.

## Feature engineering.
The three feature engineering notebooks take time to run and were the longest notebooks in terms of lines of code. I had to move some calculations to "Calculations for clicks" and "Calculations for buys" notebooks, and also moved definitions of functions, common to several feature engineering notebooks, to a dedicated notebook "OTTO common feature engineering". To further speed up the notebooks, I had to rewrite some code using polars library instead of pandas. All of this made the notebooks managable in terms of run time and complexity.

Notebooks that calculate the w2vec features for carts and orders take even more time to run, than the corresponding notebooks that calculate all the other features. So, I decided to split each of those notebooks into two notebooks, each processing its chunk of test data. Already after the competition I tried several improvements that increased feature calculation speed, but even with that improvement it takes about 3 hours to calculate the w2vec features for each chunk of the test data.

As many features are common between the notebooks, I will now provide features used at least in one of the models in a single list.
* Session history features (value is equal to some constant if candidate aid is not present in the session):
  * **n** - 0 for the last viewed aid, 1 for aid last viewed before, e.t.c, 125 for aids never viewed;
  * **time_delta** - time in seconds from a moment when aid was last viewed to the last action in session;
  * **type_last** - 0 if no buys for the aid in the session, 1 if the last buy is a cart, 2 if the last buy is an order;
  * **count_views** - number of interactions with aid in the session;
  * **time_viewed** - time from user's click on candidate aid until next event, clipped to 180 seconds and then summed for all interactions with the aid.
* Other session features (features, that only depend on session):
  * **ts_diff** - time in seconds between last event and event before last;
  * **session_time** - time in seconds from first to last event in the session (used in carts and orders models only);
  * **events_last_3hours** - total number of events last 3 hours of session (used in carts and orders models only);
  * **buys_this_session** - total number of cart/order events in session (used in carts and orders models only);
  * **history_mean** - w2vec mean similarity between last aid and previous 4 aid before that (used in carts and orders models only);
  * **buys_in_session** - 0 if no buys, 1 if only carts, 2 if at least 1 order is present in session (used in orders model only).
* Global per aid average counts:
  * **daily_aid_count** - normalized count of events with aid for the previous day;
  * **same_day_aid_count** - normalized count of events with aid for the day;
  * **aid_count_weekly** - normalized count of events/carts/orders with aid for the week;
  * **aid_counts** - total interactions with candidate aid in full sessions (used in clicks model only);
  * **aid_counts_buys** - total buys for candidate aid in full sessions (used in orders model only);
  * **aid_counts_orders**, **aid_counts_carts** - total orders/carts for candidate aid in full sessions;
  * **conv** - simple conversion rate, number of sessions with aid buied divided by total number of sessions with any event with aid (used in carts model only);
  * **total_2order_conv**, **total_2cart_conv** - feature depending on type_last feature. If aid was has no buys, here is conversion rate from views to orders, else - conversion rate from carts to orders or from orders to orders. Similar feature was constructed for carts, with click2cart, cart2cart and order2cart conversion rates (used in carts and orders models only);
  * **clicks_before_buy** - how many times on average aid is clicked before first buy (used in carts and orders models only);
  * **time_viewed_clipped** - for how long on average aid is viewed before first buy, before averaging values clipped to 180. This feature has low importance and I thought about removing it, but experiment showed results go a bit down in that case.
* Features built using co-visitation matrixes and w2vec for clicks model:
  * **wgt_matrix** - sum of co-visitaion matrix weights for the last 5 aids, using "regular" co-visitation click2click matrix (same co-visitaion matrix that was used for candidate generation);
  * **wgt_exp** - sum of co-visitaion matrix weights for the last 10 aids normalized by n (divided weight by 1 for the last aid, by 2 for aid before it, then by 3 and so on), using "experimental" co-visitation click2click matrix;
  * **wgt_last** - exact next click-to-click co-visitation matrix values for the last aid in the session;
  * **wgt_before_last** - exact next click-to-click co-visitation matrix values for the last aid in the session;
  * **similarity_first** - w2vec similarity between candidate and last aid;
  * **similarity_second** - w2vec similarity between candidate and aid before last.
* Features built using co-visitation matrixes and w2vec for carts and orders models:
  * **wgt_buy2buy** - co-visitation buy2order/buy2cart matrix feature;
  * **wgt_c2buy_short** - co-visitation click2buy matrix feature (matrix counts only cases when there is 1 hour or less between click and buy event);
  * **wgt_c2buy_full** - co-visitation click2buy matrix feature for 30 last aids (if they are within 3 hours from the last event);
  * **wgt_c2buy_6_from_full** - sum of co-visitaion matrix weights for the last 5 aids, using "regular" co-visitation click2click matrix;
  * **w2v_20_mean** - average w2vec similarity between candidate and last 20 aids (3 hours from last event);
  * **w2v_20_min** - minimal w2vec similarity between candidate and last 20 aids (3 hours from last event);
  * **w2v_5_max** - maximal w2vec similarity between candidate and last 5 aids;
  * **w2v_5_min** - minimal w2vec similarity between candidate and last 5 aids (this feature also has low importance, but its removal decreased result a bit).

## Training the GBDT models and predicting
Notebooks training the GBDT models are the only ones that use GPU. I tried both catboost and LGBM models, and LGBM showed better results. To produce the final prediction for clicks and carts only LGBM predictions are used, while for orders I've build two cross validation datasets and used one of them to train LGBM model, and another one - to train catboost model. Then I combined predictions made by two models, and found out that it mamkes a slightly better prediction than a single prediction made by LGBM model. I kept that pipeline with two models and two cross-validation datasets for orders, but decided against implementing a similar pipeline for carts.

I only used sessions with at least one postive candidate generated. This means, that number of sessions used to train carts or orders models was relatively low, as for these models more than 80% of sessions do not have any positive candidates, and even less sessions have at least a single candidate guessed at candidate generation stage. So, for these models I didn't have significant memory problems. I did decrease number of negative examples for those models, but that was mostly beacuse it increased model's performance.

The situation with clicks model was different. Almost all the sessions have positive targets, and about 60% of them have correct candidates selected at candidate generation stage, so the clicks model has times more data compared to the carts and orders models. I kept removing more negative candidates even after this started decreasing the model's performance. I converted all the float variables to float16 after loading the parquet file. I wrote a custom function to split the cross-validation dataset into folds. But anyway, low memory available for kaggle notebooks with GPU was a limitation for the model.

Close to competition's final days, I have increased number of candidates for carts and orders models from 50 to 75. This turned out to be a disappointment as result of final prediction didn't improve even a little bit. But more candidates means more time to run the feature engineering notebooks and more memory consumed by notebooks. After increasing number of candidates I had to move prediction for carts and orders models to a separate notebook, as a notebook with GPU support started falling with memory error.

## Final formatting and submitting the results
Little can be said abouth this notebook. I wrote the code in a way that it was possible to upload results for a single model or results for all the three models, to track improvements for each model on leaderboard.

## Summary and what could have been done better
I started working on the project about a month after the competition was launched and still managed to get into top 3% participants, well in the middle of the silver zone. That should be considered to be a fairly good result. After the competition, I've read all the posts of top teams members and understood that all of them used servers with way more RAM and GPU available. I worked solo and lost a competition to people who mostly worked in teams and had way more computational resourse. If all the time I spent trying to fit all the data into available memory could be spent on running additional experiments, I would have been be able to produce a better result.

Having said that, no doubt that even with the resourse available I could have done better. I had enough memory to add much more features for carts and orders models. I could have run more notebooks in parallel and tried even more features, more types of co-visitation matrixes, more versions of w2vec models. I could have improved my carts and orders models by using some features I've only used for clicks model. For example, I haven't even tried features built for exact last aid in session for carts and orders models, and I haven't tried features built with any of click2click models for carts/orders. Then, there were several bugs in code that also a bit decreased the result. Then, there were all sorts of original co-visitation matrixes used by top teams, like a matrix that only counts first few aids in session, a matrix that counts last week's results, matrixes that separately count events before/after 2 PM. If I had a bit more time to think, probably I could have come to some of this ideas. Probably one of my mistakes was always trying to add additional counts instead of thinking about additional types of co-visitation matrixes.

## March 2023 upload to github.
In March 2023, about a month after the competition, I decided to review the code, add some comments, delete commented unused code and upload the notebooks to github. At this point further improving recommendations was not my goal anymore. But while adding comments and reviewing the code I couldn't help making some changes. I fixed a few bugs, moved addidional fucntions to otto_common notebook, created a separate notebook with functions common to feature engineering, checked for ways to speed up the word2vec feature notebook, removed a few features that actually decreased model's performance, e.t.c. As a result not just the code became shorter and clearer, but also the result have improved.
