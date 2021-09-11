# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4: Group Project


## Overview ##
An Instagram post can be divided into two components – image and non-image factors.

On the image side of things, I seek to find the optimal image ingredients, henceforth known as props, to maximise each post’s performance. On the non-image side of things, I seek to find the optimal caption and post datetime to maximise each post’s performance as well. 

I use the term post performance instead of simply aiming for a high like number because a post’s like_count is heavily dependent on the poster’s following_count. Therefore, by maximising post performance, I mean maximising a post’s like count given the poster’s following count. 

## Background ##
Instagram Marketing is an important constituent of many companies’ marketing strategy. Besides it’s sister placement, facebook, it is common hygiene to buy advertisments that gives the most cost effective CPM (which is cost per 1000 views), CPC (which is cost per click), and CPResult (cost per result)

Having a well established IG page not only generates a strong following, it helps companies further engage non-followers by allowing stronger like-audiences to further lower CPC by giving the FB algorithm more data.

The First step to establishing a strong IG page is getting likes. Likes signals to users others’ sentiments toward a company and helps get a page recommended on IG explore.

Likes also help a pages’ posts get more viewability as the IG algorithm increases the likelihood of a post by the page being seen on IG feed when the user has liked the post from the page.

Likes are so important that many companies actually use paid ads with the campaign objective “engagement” to boost engagements to their posts
While paid engagement is effective and provides short term boosts, growing organic likes is always more cost effective in the long run as there are no advertising costs to generate this long-term benefit


## Problem Statement ##
Companies often post Instagram photos that do not perform well. Some resort to paid advertisements but, as mentioned earlier, it is only a short term measure.

Others spend exorbitant amounts of money hiring marketing agencies to handle each and every Instagram post within each campaign. Therefore, in this project, I seek to find the optimal image props, caption, and post datetime to maximise each post’s performance.

## Methodology ##
With our goal in mind, I start with scoping the project. I picked the gaming and computer hardware industry because they are the most active industry that advertises on social media in Singapore. Next, we go into data collection, then we work on our first model, which process non-image data. After that, we process our image data in a second model. Lastly we compile the insights from both models.


## Data dictionary ##

### Non-Image Model ###
1) Account: Poster's Instagram name
2) User Name: Poster's Instagram handle
3) Followers at Posting: Amount of followers recorded at post's time of posting
4) Post Created: Date and time of post's creation
5) Post Created Date: Date of post's creation
6) Post Created Time: Time of post's creation
7) Type: Type of post - (1) album, (2) photo, (3) IGTV, (4) video
8) Total Interaction: Number of Likes + Comments
9) Likes: Number of Likes
10) Comments: Number of Comments
11) Views: View count of post (only applicable to IGTV and Videos which we remove during cleaning)
12) Like and View Counts Disabled: True or False; if the post's like and view count are disabled
13) URL: Link to post
14) Link: Link to post
15) Photo: Link to photo
16) Description: Post's Caption
17) Overperforming: binary variable, if post is deemed by my dataset provided as having overperformed. This is measured by Likes and Followers at Posting.

### Image Model ###
1) Prop_1: Prop that Resnet50 predicts, with the highest probability, is present in the image.
2) Prop_2: Prop that Resnet50 predicts, with the second highest probability, is present in the image.
3) Prop_3: Prop that Resnet50 predicts, with the third highest probability, is present in the image.
4) Prop_4: Prop that Resnet50 predicts, with the fourth highest probability, is present in the image.
5) Prop_5: Prop that Resnet50 predicts, with the fifth highest probability, is present in the image.
6) Prop_6: Prop that Resnet50 predicts, with the sixth highest probability, is present in the image.
7) Prop_7: Prop that Resnet50 predicts, with the seventh highest probability, is present in the image.
8) Prop_8: Prop that Resnet50 predicts, with the eighth highest probability, is present in the image..
9) Prop_9: Prop that Resnet50 predicts, with the ninth highest probability, is present in the image.
10) performance: Like number divided by account's follower count
11) Like_number: number of likes received by the post
12) brand: post's company
13) Overperforming : All posts with performance greater than the median (0.008101) were classified as Overperforming = 1; binary variable.

## Brief Summary of Analysis ##

### Non-Image Model ###
The dataset consists of posts from @aftershockpc, @ergotunechair, @logitechg, @msi, @prismplusdisplays, @razer, @secretlab, @theomnidesk, and @steelseries since September 2020. As to where I got it from, I’m not at liberty to say.

#### EDA ####
In the EDA phase, I found that, on captions, there were no clear distinctions between phrases in posts that overperform or otherwise. But, this is likely due to the distribution of the dataset as some accounts post far more than others. For example, prismplus’ brand name appears as the top phrase in both the overperforming and underperforming posts. Further, the low over frequency of each phrase in underperforming posts also reveal a very low count of underperforming posts as compared to overperforming posts. This indicates that we will need some hardcore stratifying when it comes to modelling.

On the top 20 post dates there seems to be no discernible pattern. I looked into each date in the diagram and they were not dates with any particular significance. On the top 20 post times, a pattern emerged – most overperforming posts tend to be posted at 6pm and 12pm.

Something peculiar I found in the accounts. When looking at the proportion or percentage of overperforming post in each account, ergotune fairs the worst among the 9. This is strange because ergotune’s Instagram advertisements are directed to the most number of users in the list. This further proves the earlier point that just blindly spending money on Instagram advertisements is not the solution.

#### Modelling and Hyperparameter Tuning ####
I formed the model with features (1) Description, (2) Date Time, (3) User Name, and (4) Overperforming (binary target variable).
After performing hyperparameter Tuning, I found that the best word vectorizer for 'Description' was TFIDF and the best model for our features was Logistic Regression. I therefore used an optimised TFIDF-Logistic-Regression model. 

### Image Model ###
Like before, we collected data on the same 9 companies. Using Selenium, I scraped posts launched within the last 1 month from each User Name. I then ran the images through Resnet50 which returned 9 predictions of items or props within each image. These predictions are ranked in terms of probability of their being present in the image, we can therefore take that as prop prominence in the image -- I assume that if there is a higher chance that item A is in image X, it translates to item A being a more prominent prop in image X. I then matched each set of predictions to the like numbers that correspond to the image post they were taken from. 

#### EDA ####
Since I did EDA on accounts and post date and time in the non-image model, I focused on prop eda. We see that for the most prominent prop, most overperforming posts feature a website. For the second most prominent prop, it’s a monitor.

#### Modelling and Hyperparameter Tuning ####
I formed a model with features Prop_1 to Prop_9 and Overperforming (Target variable). After using pycaret, I found the best performing model to be GaussianNB, however, as pycaret only supports model interpretation for tree-based classifiers, I went with xgboost, the next best model.

## Findings and Recommendations ##

### Non-Image Model ###
I found that these phrases were the most impactful in determining an overperforming post: 'battlestation', 'ergotune', 'aftershockpc', 'g733', 'g915tkl', 'wfh', 'gamingkeyboard', 'pcgaming', 'keyboard' ,' g305', 'valorant', 'feel its', 'giveaway', 'take', 'color', 'www', 'pc', 'tag', 'clean', 'screen', 'is setup', 'than', 'sit', 'many', 'looking could'.

On the Instagram accounts, previously in the EDA phase, we saw that ergotune was at the bottom of the performance list in terms of percentage of overperforming posts. Now, however, we find that the word ergotune is a significant feature in predicting overperforming posts. Conversely, quoting MSI and standing desk, the latter being associated the Omnidesk, pushes our model to predict an underperforming post.

On buzzwords, we see work from home and giveaway. These two definitely stand out as the more expected buzzwords among the others given that most products sold by our 9 companies function as work from home tools and also because everyone is a sucker for giveaways.

The last thing to note is that the model also processed each post’s date and time stamp, yet neither appeared as important feature, this implies that it is possible that the marketing industry’s fixation on an optimal post date and time could really just be bro science or tradition with no true efficacy.


### Image Model ###
Using SHAP value, we revealed props that prompted the model to classify it as overperforming and those that prompted the model to classify is underperforming. However, due to instances where a given prop (say, notebook) appears under a different prop column (Prop_6, Prop_9, and Prop_7), there were instances where props were classifed as both prompting a model to classify it as overperforming and under performing. Case in point: notebook. Prop_6_notebook and Prop_9_notebook significantly prompted the model to classify the image as overperforming, however Prop_7_notebook significantly prompted the model to classify the image as underperforming.

## Limitations ##
For Non-Image Data, the Dataset being extremely skewed toward overperforming posts forced us to resort to oversampling through smote. This process definitely increased my scores tremendously, however, the inorganic sampling method leaves us with the question of whether we could have gotten more desirable results if we had a more balanced dataset and did not need to use smote.

For the Image Data, in trying to account for a prop’s prominence, I caused results being suboptimal. While I did manage to sift out some important props, the few contradictions were impossible to fix.


## Future Steps ##
For Non-Image Data, I believe that if we do more comprehensive industry analysis of Instagram profiles, we will be able to manually put together a dataset that has a balance between overperforming and underperforming posts.

For Image Data, in the future, we definitely should discount prop prominence. Instead, we should merge the props to form an all text row and perform NLP classification modelling.

I believe that after making the above 2 changes, we should move toward forming a recommendation engine that can recommend props and captions based on each company’s (and their competitors') historical posting pattern. 

Lastly, on industry use, while admittedly, this project isn’t quite near the implementation stage, I believe it’s at the very least, close to a proof of concept. Let’s call it a strong suggestion of concept. Companies in different industries need only to scrape the data of their own and their competitors’ Instagram posts and they can immediately be fed with keywords and props that will likely improve their Instagram performance.

An important thing to note however, is that this project is definitely not meant to replace creatives. Even if we were able to produce an optimised model that accounted for all the limitations previously stated, every Instagram post made by a company will always need a creative mind to piece together the suggestions from our models. Ultimately, it is but a tool to aid the creative process.
