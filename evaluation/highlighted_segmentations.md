# Notes

The following table highlights some of the segmentations from each category,
where instace F1 score is bigger for metric version 2 (based on segments and their position in word) and metric  
verison 3 (based on breaks only). We can see that the first metric scores low when there is more  
segments predicted than there is real segments.


__________________


| category | word                  | 	segmentation             | prediction                   | 	distance | overlaps | f1v2_tp | f1v2_fp | f1v2_fn | f1v3_tp | f1v3_fp | f1v3_fn | f1v2_score | f1v3_score | abs_diff |
|----------|-----------------------|---------------------------|------------------------------|-----------|----------|---------|---------|---------|---------|---------|---------|------------|------------|----------|
| 001      | grubstake             | grub#stake	               | gr#ub#sta#ke                 | 2         | 0        | 0       | 4       | 2       | 1       | 2       | 0       | 0.0        | 0.5        | 0.5      |
| 001      | giga-joule            | giga#joule                | gig#a#jo#ule	                | 2         | 0        | 0       | 4       | 2       | 1       | 2	      | 0	      | 0.0        | 0.5	       | 0.5      |
| 001      | 	5-hydroxytryptamine	 | 5#hydroxy#tryptamine      | 	5#hydro#xy#try#pta#mine     | 	3        | 	1       | 	1      | 	5      | 	2      | 	2      | 	3      | 	0      | 	0.222     | 	0.571     | 	0.349   |
| 010      | 	neurofunctionally    | 	neuro#function#al#ly     | 	ne#uro#fu#nction#ally       | 	3        | 	0       | 	0      | 	5      | 	4      | 	2      | 	2      | 	1      | 	0.0       | 	0.571     | 	0.571   |
| 010      | 	neurophysiopathology | 	neuro#physio#pathology   | 	ne#uro#phy#sio#path#ology   | 	3        | 	0       | 	0      | 	6      | 	3      | 	2      | 	3      | 	0      | 	0.0       | 	0.571     | 	0.571   |
| 011      | 	tetraethyllead       | 	tetra#ethyl#lead         | 	te#tra#eth#yl#lea#d         | 	3        | 	0       | 	0      | 	6      | 	3      | 	2      | 	3      | 	0      | 	0.0       | 	0.571     | 	0.571   |
| 011      | 	brokenhanded         | 	broke#n#handed           | 	broken#hand#ed              | 	2        | 	0       | 	0      | 	3      | 	3      | 	1      | 	1      | 	1      | 	0.0       | 	0.5       | 	0.5     |
| 100      | 	'fridges             | 	'fridge#s                | 	'#fridge#s                  | 	1        | 	1       | 	1      | 	2      | 	1      | 	1      | 	1      | 	0      | 	0.4       | 	0.667     | 	0.267   |
| 100      | 	ECECs                | 	ecec#s                   | 	ec#ec#s                     | 	1        | 	1       | 	1      | 	2      | 	1      | 	1      | 	1      | 	0      | 	0.4       | 	0.667     | 	0.267   |
| 101      | 	flangeways           | 	flange#way#s             | 	fl#ange#ways                | 	2        | 	0       | 	0      | 	3      | 	3      | 	1      | 	1      | 	1      | 	0.0       | 	0.5       | 	0.5     |
| 101      | 	haut-pas             | 	haut#pa#s                | 	ha#ut#pas                   | 	2        | 	0       | 	0      | 	3      | 	3      | 1       | 	1      | 	1      | 	0.0       | 	0.5       | 	0.5     |
| 110      | 	transcendentalists   | 	transcend#ent#al#ist#s   | 	trans#cend#ental#ists       | 	3        | 	0       | 	0      | 	4      | 	5      | 	2      | 	1      | 	2      | 	0.0       | 	0.571     | 	0.571   |
| 110      | 	chemolithoautotrophs | 	chemo#litho#auto#troph#s | 	che#mo#lit#ho#au#to#tro#phs | 	5        | 	0       | 	0      | 	8      | 	5      | 	3      | 	4      | 	1      | 	0.0       | 	0.545     | 	0.545   |
| 111      | 	newsboards           | 	new#s#board#s            | 	news#boards                 | 	2        | 	0       | 	0      | 	2      | 	4      | 	1      | 	0      | 	2      | 	0.0       | 	0.5       | 	0.5     |
| 111      | 	rollerballs          | 	roll#er#ball#s           | 	roller#balls                | 	2        | 	0       | 	0      | 	2      | 	4      | 	1      | 	0      | 	2      | 	0.0       | 	0.5       | 	0.5     |

## Number of predicted vs. real segments for categories

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

## occurrences where less breaks are predicted and f1 is still bigger

We can see that ver3 metric still does better even if less segments are predicted  
below are some examples from different categories.

| category | word               | 	segmentation           | prediction             | 	distance | overlaps | f1v2_tp | f1v2_fp | f1v2_fn | f1v3_tp | f1v3_fp | f1v3_fn | f1v2_score  | f1v3_score | abs_diff |
|----------|--------------------|-------------------------|------------------------|-----------|----------|---------|---------|---------|---------|---------|---------|-------------|------------|----------|
| 001	     | counter-password   | 	counter#pass#word      | 	counter#password      | 	1        | 	1       | 	1	     | 1	      | 2       | 	1      | 	0      | 	1      | 	0.4	       | 667	       | 267      |
| 010	     | actionally         | 	act#ion#al#ly          | 	action#ally           | 	2	       | 0        | 	0      | 	2      | 	4      | 	1      | 	0      | 	2      | 	0.0        | 	0.5       | 	0.5     |
| 011	     | tenderhearted      | 	tend#er#heart#ed       | 	tender#hearted        | 	2	       | 0	       | 0	      | 2	      | 4	      | 1	      | 0	      | 2	      | 0.0	        | 0.5        | 	0.5     |
| 101	     | battlegrounds      | 	battle#ground#s        | 	battle#grounds        | 	1        | 	1       | 	1      | 	1	     | 2       | 	1      | 	0	     | 1       | 	0.4	       | 667        | 	267     |
| 110	     | transcendentalisms | 	transcend#ent#al#ism#s | 	trans#cend#ental#isms | 	3        | 	0	      | 0	      | 4	      | 5	      | 2       | 	1	     | 2	      | 0.0	571	571 |

## comparing original f1 metric with f1_ver2

At first it looked like original f1 metric is scoring worse than proposed f1_ver2 metric.  
That seemed strange. In fact it was wrong because original metric was not implemented 
correctly after changes i made to the code. It was counting excluded segmentations in a way that  
penalised the metric significantly. At */evaluation/ver1_ver2_diff.csv* is a table of all instances  
where original f1 scores better because of mismatched segments.



### English - BERT Tokenizer (uncased)

Here is a table for better understanding of word categorisation.

| category | inflection | derivation | compound | precision | recall | f_measure | f1_ver2 | f1_ver3 | distance |
|:--------:|:----------:|:----------:|:--------:|:---------:|:------:|:---------:|:-------:|:-------:|:--------:|
|    000   |     no     |     no     |    no    |   3.43    |  9.43  |   5.03    |  5.04   |  6.64   |   1.75   |
|    001   |     no     |     no     |    yes   |   53.09   | 66.79  |   59.16   |  60.25  |  68.28  |   0.84   |
|    010   |     no     |     yes    |    no    |   19.54   | 26.18  |   22.38   |  35.28  |  46.75  |   2.70   |
|    011   |     no     |     yes    |    yes   |   30.17   | 28.45  |   29.29   |  49.16  |  66.74  |   2.19   |
|    100   |     yes    |     no     |    no    |   11.23   | 17.68  |   13.73   |  16.70  |  22.43  |   2.62   |
|    101   |     yes    |     no     |    yes   |   45.50   | 43.64  |   44.55   |  53.91  |  67.40  |   1.43   |
|    110   |     yes    |     yes    |    no    |   16.21   | 17.03  |   16.61   |  31.05  |  47.37  |   3.27   |
|    111   |     yes    |     yes    |    yes   |   25.04   | 20.04  |   22.27   |  36.79  |  57.98  |   3.12   |
|    all   |      -     |      -     |     -    |   16.72   | 22.83  |   19.30   |  27.65  |  36.78  |   2.55   |