# kaggle_OTTO
## Task and solution overview.
This is the code for kaggle competition called "OTTO – Multi-Objective Recommender System". OTTO is Germany's largest online retailer. The task is to predict, which exact next item user is going to click next and which items user is going to add to cart or order before the end of the test period. The competition data is real life new users' sessions on OTTO website. The test dataset includes one week of users' sessions, truncated at a random point. Organizers also provide participants with history of full user's sessions for four weeks, preceding the test period. But no metainformation is available for all the items, that show up in both datasets. Participants only have item ID's, that are called AIDs, and there are about 1.8 mln AIDs showing up in the competition data, including both full and truncated user sessions.

From the beginning I have decided to use Jupyter notebooks ran on kaggle website to produce the solution. However, this decision turned out to have a number of complications. Firstly, kaggle notebooks without GPU support at a time of competition had RAM limit of 30 Gb and I had to spend some time, for example, writing code that would merge two dataframes chunk by chunk, as using a simple merge would cause a memory error. Kaggle notebooks with GPU available have RAM limit of just 13 Gb, that made me choose between some features, instead of using all of them. Then, Jupyter notebooks are not that useful when dealing with projects that require a complicated data pipeline. Total number of notebooks used to produce the final solution is 29, and that number does not include a few more notebooks used to test some approaches that brought no fruit.

The solution's pipeline include the following major stages:
* creating a cross-validation dataset from the last week of known full sessions, similar to the test dataset;
* calculated co-visitation matrixes, word2vec models and making some other side calculations aside of the main pipeline;
* generating candidates;
* engineering features;
* training the GBDT re-ranking models on the cross-validation dataset and using those models to select most relevant candidates, generated for the test sessions;
* some final formatting and submitting the results.
Generating candidates, engineering features, training models and making final prediction stages are separate for each prediction, i.e. for clicks, carts and orders. For each of these predictions, a dedicated model is being trained on a different dataset and for different candidates, although many features are actually common.

## A closer look at input data and metric.
