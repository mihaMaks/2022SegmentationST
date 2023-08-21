# WORD SEGMENTATION - metric comparison
New comparison is needed when we filter the data again, this was done based on how many words are   
surface-level segmented !!!


This file is summary of comparison between 3 F1-score metrics that are scoring  
how well pretrained BERT-tokenizer predicts segemetation of words.  
## Introduction

There are 3 versions of metrics that have different scoring methods that can be summarized:

 - f1_ver1: this is original metric used by *SIGMORPHON(how to name it?)*  
    This metric compares each segment in ground-truth segmentation and predicted segmentation with each other

- f1_ver2: compares segments and their position in segmentation

- f1_ver3: compares position of 'breaks'/separations in segmentation in word

Based on methods that metrics mentioned above use to calculate F1 score we can predict that metric ver1 and ver2  
are very similar with a note that ver2 is stricter as it also compares position of segments
From that point of view we could say that ver3 is lose in its scoring as it doesn't 'care' for letters in segments.
It views segments only as objects with different positions in segmentation.
__________________

## Data overview
Before we jump into analytics i want to mention that data used for comparison was dirty so some cleaning was needed
Below is table that shows hom much noise was present and how data is balanced base on number of predicted segments vs. ground segmentations

| category | all segmentations |   excluded    | more pred than real |     same      | less pred than real |
|:--------:|:-----------------:|:-------------:|:-------------------:|:-------------:|:-------------------:|
|   000    |       8615        |   11 (0,1%)   |    7792 (90,5%)     |  812 (9,5%)   |      0 (0.0)%       |
|   001    |       2009        |   34 (1.7%)   |     763 (37,9%)     | 1167 (58.1%)  |      46 (2.3%)      |
|   010    |       21018       | 7777 (37,0%)  |    8207 (39,0%)     | 4038 (19,2%)  |     996 (4,8%)      |
|   011    |        762        |  314 (41,2%)  |      69 (9,1%)      |  182 (23,9%)  |     197 (25,8%)     |
|   100    |       12086       | 2327 (19,3%)  |    7004 (58,0%)     | 2423 (20,0%)  |     332 (2,7%)      |
|   101    |       1421        |  253 (17,8%)  |     141 (10,0%)     |  677 (47,6%)  |     350 (24,6%)     |
|   110    |       11117       | 5576 (50,2%)  |    2141 (19,3%)     | 2105 (18,9%)  |    1295 (11,6%)     |
|   111    |        343        |  151 (44,0%)  |      14 (4,1%)      |  40 (11,7%)   |     138 (40,2%)     |
|   all    |      57371        | 16443 (28,7%) |    26130 (45,6%)    | 11444 (19,9%) |     3354 (5,8%)     |

## Table of metric performances

Table also shows how words are categorised.

| category | inflection | derivation | compound | precision | recall | f_measure | f1_ver2 | f1_ver3 | distance |
|:--------:|:----------:|:----------:|:--------:|:---------:|:------:|:---------:|:-------:|:-------:|:--------:|
|   000    |     no     |     no     |    no    |   3.44    |  9.44  |   5.04    |  5.04   |  6.64   |   1.75   |
|   001    |     no     |     no     |    yes   |   54.21   | 67.96  |   60.31   |  60.31  |  68.33  |   0.84   |
|   010    |     no     |     yes    |    no    |   30.81   | 43.84  |   36.19   |  36.19  |  47.37  |   2.70   |
|   011    |     no     |     yes    |    yes   |   51.20   | 47.73  |   49.40   |  49.40  |  66.63  |   2.19   |
|   100    |     yes    |     no     |    no    |   13.73   | 21.83  |   16.86   |  16.80  |  22.55  |   2.62   |
|   101    |     yes    |     no     |    yes   |   55.63   | 52.72  |   54.14   |  54.14  |  67.59  |   1.43   |
|   110    |     yes    |     yes    |    no    |   31.17   | 33.89  |   32.48   |  33.42  |  48.40  |   3.27   |
|   111    |     yes    |     yes    |    yes   |   42.56   | 34.50  |   38.11   |  38.11  |  58.18  |   3.12   |
|   all    |      -     |      -     |     -    |   23.48   | 34.84  |   28.05   |  28.02  |  36.78  |   2.55   |
From: https://github.com/mihaMaks/2022SegmentationST/blob/main/evaluation/evaluation.txt


## comparing f1_ver1 and f1_ver2 metric

As mentioned before metric ver2 and ver3 are similar in their approaches. From the table above it can bee seen that  
ver2 is more strict than ver1. For that reason our comparison will focus on relationship between metric ver2 and ver3.
Below are some examples why ver2 metric is better and why its approach makes more sense.
We are trying to illustrate how ver1 metric can be exploited by some segmentations.

| category | word          | 	segmentation   | prediction       | 	distance | f1v1_tp | f1v1_fp | f1v1_fn | f1v2_tp | f1v2_fp | f1v2_fn | f1v1_score | f1v2_score | abs_diff |
|----------|---------------|-----------------|------------------|-----------|---------|---------|---------|---------|---------|---------|------------|------------|----------|
| __010	   | erter	        | ert#er	         | er#ter           | 	2        | 	0      | 	2      | 	2      | 	1      | 	1      | 	1      | 	0.0    | 	0.5       | 	0.5       |
| __010    | 	unlyrically  | 	un#lyric#al#ly | 	un#ly#rica#lly  | 	4        | 	1      | 	3      | 	3      | 	2      | 	2      | 	2      | 	0.25   | 	0.5       | 	0.25      |
| __110    | 	recorrecting | 	re#correct#ing | 	rec#or#re#cting | 	5        | 	0      | 	4      | 	3      | 	1      | 	3      | 	2      | 	0.0    | 	0.286     | 	0.286     |
From: https://github.com/mihaMaks/2022SegmentationST/blob/main/evaluation/ver1_ver2_diff.csv

As there is only around 30 instances of this error in our data set the difference in score is minimal.

## comparing f1_ver2 and f1_ver3 metric

By looking at table of results the difference in ver2 and ver3 is obvious. Table below shows that metric ver3
is not punished hard if there is more predicted segments. The model used is biased in that way as it predicts more segments
than it should in about 45% of instances.


| category | word                  | 	segmentation             | prediction                   | 	distance | f1v2_tp | f1v2_fp | f1v2_fn | f1v3_tp | f1v3_fp | f1v3_fn | f1v2_score | f1v3_score | abs_diff |
|----------|-----------------------|---------------------------|------------------------------|-----------|---------|---------|---------|---------|---------|---------|------------|------------|----------|
| 001      | grubstake             | grub#stake	               | gr#ub#sta#ke                 | 2         | 0       | 4       | 2       | 1       | 2       | 0       | 0.0        | 0.5        | 0.5      |
| 001      | giga-joule            | giga#joule                | gig#a#jo#ule	                | 2         | 0       | 4       | 2       | 1       | 2	      | 0	      | 0.0        | 0.5	       | 0.5      |
| 001      | 	5-hydroxytryptamine	 | 5#hydroxy#tryptamine      | 	5#hydro#xy#try#pta#mine     | 	3        | 	1      | 	5      | 	2      | 	2      | 	3      | 	0      | 	0.222     | 	0.571     | 	0.349   |
| 010      | 	neurofunctionally    | 	neuro#function#al#ly     | 	ne#uro#fu#nction#ally       | 	3        | 	0      | 	5      | 	4      | 	2      | 	2      | 	1      | 	0.0       | 	0.571     | 	0.571   |
| 010      | 	neurophysiopathology | 	neuro#physio#pathology   | 	ne#uro#phy#sio#path#ology   | 	3        | 	0      | 	6      | 	3      | 	2      | 	3      | 	0      | 	0.0       | 	0.571     | 	0.571   |
| 011      | 	tetraethyllead       | 	tetra#ethyl#lead         | 	te#tra#eth#yl#lea#d         | 	3        | 	0      | 	6      | 	3      | 	2      | 	3      | 	0      | 	0.0       | 	0.571     | 	0.571   |
| 011      | 	brokenhanded         | 	broke#n#handed           | 	broken#hand#ed              | 	2        | 	0      | 	3      | 	3      | 	1      | 	1      | 	1      | 	0.0       | 	0.5       | 	0.5     |
| 100      | 	'fridges             | 	'fridge#s                | 	'#fridge#s                  | 	1        | 	1      | 	2      | 	1      | 	1      | 	1      | 	0      | 	0.4       | 	0.667     | 	0.267   |
| 100      | 	ECECs                | 	ecec#s                   | 	ec#ec#s                     | 	1        | 	1      | 	2      | 	1      | 	1      | 	1      | 	0      | 	0.4       | 	0.667     | 	0.267   |
| 101      | 	flangeways           | 	flange#way#s             | 	fl#ange#ways                | 	2        | 	0      | 	3      | 	3      | 	1      | 	1      | 	1      | 	0.0       | 	0.5       | 	0.5     |
| 101      | 	haut-pas             | 	haut#pa#s                | 	ha#ut#pas                   | 	2        | 	0      | 	3      | 	3      | 1       | 	1      | 	1      | 	0.0       | 	0.5       | 	0.5     |
| 110      | 	transcendentalists   | 	transcend#ent#al#ist#s   | 	trans#cend#ental#ists       | 	3        | 	0      | 	4      | 	5      | 	2      | 	1      | 	2      | 	0.0       | 	0.571     | 	0.571   |
| 110      | 	chemolithoautotrophs | 	chemo#litho#auto#troph#s | 	che#mo#lit#ho#au#to#tro#phs | 	5        | 	0      | 	8      | 	5      | 	3      | 	4      | 	1      | 	0.0       | 	0.545     | 	0.545   |
| 111      | 	newsboards           | 	new#s#board#s            | 	news#boards                 | 	2        | 	0      | 	2      | 	4      | 	1      | 	0      | 	2      | 	0.0       | 	0.5       | 	0.5     |
| 111      | 	rollerballs          | 	roll#er#ball#s           | 	roller#balls                | 	2        | 	0      | 	2      | 	4      | 	1      | 	0      | 	2      | 	0.0       | 	0.5       | 	0.5     |
From: https://github.com/mihaMaks/2022SegmentationST/blob/main/evaluation/ver2_smaller_ver3_diff.csv

But there are still instances where ver3 metric scores better even if less segments are predicted.

| category | word               | 	segmentation           | prediction             | 	distance | f1v2_tp | f1v2_fp | f1v2_fn | f1v3_tp | f1v3_fp | f1v3_fn | f1v2_score | f1v3_score | abs_diff |
|----------|--------------------|-------------------------|------------------------|-----------|---------|---------|---------|---------|---------|---------|------------|------------|----------|
| 001	     | counter-password   | 	counter#pass#word      | 	counter#password      | 	1        | 	1	     | 1	      | 2       | 	1      | 	0      | 	1      | 	0.4	      | 0.667	     | 0.267    |
| 010	     | actionally         | 	act#ion#al#ly          | 	action#ally           | 	2	       | 	0      | 	2      | 	4      | 	1      | 	0      | 	2      | 	0.0       | 	0.5       | 	0.5     |
| 011	     | tenderhearted      | 	tend#er#heart#ed       | 	tender#hearted        | 	2	       | 0	      | 2	      | 4	      | 1	      | 0	      | 2	      | 0.0	       | 0.5        | 	0.5     |
| 101	     | battlegrounds      | 	battle#ground#s        | 	battle#grounds        | 	1        | 	1      | 	1	     | 2       | 	1      | 	0	     | 1       | 	0.4	      | 0.667      | 	0.267   |
| 110	     | transcendentalisms | 	transcend#ent#al#ism#s | 	trans#cend#ental#isms | 	3        | 0	      | 4	      | 5	      | 2       | 	1	     | 2	      | 0.0	       | 0.571      | 0.571    |


We can see that ver2 is in fact very strict as one misplaced segment can impact scoring on other correctly placed segments
This is good if we would work with data that is not properly cleaned up as metric ver2 is (hard to/imposible?)
to cheat. Ver3 is fast but it is not reliable if data is not cleaned up properly.

Here are two instances where nuber of segments predicted is the same as ground segments.

| category | word               | 	segmentation        | prediction           | 	distance | f1v2_tp | f1v2_fp | f1v2_fn | f1v3_tp | f1v3_fp | f1v3_fn | f1v2_score | f1v3_score | abs_diff |
|----------|--------------------|----------------------|----------------------|-----------|---------|---------|---------|---------|---------|---------|------------|------------|----------|
| __010    | reinthrone         | 	re#in#throne        | 	rein#th#rone        | 	2        | 	0      | 	3      | 	3      | 	1      | 	1      | 	1      | 	0.0       | 	0.5       | 	0.5     |
| __011    | 	feeble-mindedness | 	feeble#mind#ed#ness | 	fee#ble#minded#ness | 	2        | 	1      | 	3      | 	3      | 	2      | 	1      | 	1      | 	0.25      | 	0.667     | 	0.417   |
From: https://github.com/mihaMaks/2022SegmentationST/blob/main/evaluation/ver2_eqSeg_ver3.csv

Because of difference in approach ver2 scores lower. What is illustrated here is how ver2 counts segments
and ver3 counts brakes which means there is always one more segment than there are breaks.
This also contributes to difference in scores and makes the two metrics harder to compare.
We fixed this problem for monomorphs (words with one segment), otherwise ver3 would ignore them  
because there is no brakes in them. That is also why scores in 000 category are much closer.

There are no instances where ver2 scores better than ver3.

## Conclusion

Metrics are different in approaches so it is hard to compare them directly.  
Both metrics have advantages and disadvantages:
- ver2: good for messy data, is too strict for cleaned data
- ver3: simple and fast, not rigorous enough, bad for messy data

What can we change moving forward?
Maybe we can find a way to punish metric ver3 more for mispredicted segments?
Can we change ver2 so that it returns rough estimation how maouch of segment/-s are correctly
predicted?



 