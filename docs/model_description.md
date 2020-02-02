# Overview:

## 1. Input and output data description

## 2. Model description

## 1. Input and output data description

### Input format: (Batch_size = None, Window_length = 6, Feature_num = 70)

Batch_size refers to the input batch size, which is variable.

Window_length refers to the number of windows included in the input. 
One window includes all the company incidents happened in six months, and the six windows are consecutive. 
By six, every input can include three years of incidents in a company.

Feature_num refers to the number of features included in every output. In this model, the 70 dimensions of features could be broken into eight parts:
*	Funding amount: 1 dimension, the normalized funding amount in the given round of funding.
* Funding round: 15 dimensions, the round of the funding in one hot encoding.
*	News: 2 dimensions, the number of reported news related to the selected company and the sentiment of the news.
*	Number of investors: 1 dimension, the number of investors involved in the investment to the selected company, if any.
*	Sectors: 19 dimensions, the sector which the selected company belongs to. One hot encoding.
*	First-tier city: 1 dimension, whether the selected company headquarter located in first-tier city in China.
*	Top investors: 28 dimensions, 28 investors with the most investments followed by further investments. One hot encoding.

### Output format: (Batch_size, Outcome)

Batch_size refers to the input batch size, which is variable.

Outcome refers to the result of funding for the selected company. 
1 for there is a funding, 0 for there is no funding. 
The “funding” here refers to the funding in the round above b, IPO and strategic investment.

The output and input are related. As the example shows below, the index for the “batch_size” in both output and input refers to the same company, meanwhile the output in some window refers to the result happened in the next window of the input.

Input (Row 1) & Output (Row 2)
|Company A	|No fund	|No fund	|Fund	|No Fund	|No Fund	|Fund	|No Fund	|No Fund|
|---|---|---|---|---|---|---|---|---|
|Company A	|0	|1	|0	|0	|1	|0	|0	|N/A|


## 2. Model Description

The applied model for the startup prediction is Temporal Convolutional Neural Network, 
TCN in abbreviation. It is a CNN model for time series problem.
The implementation code is listed below:

```
i = Input(batch_shape=(None, 6, 70))
o = TCN(nb_filters=128,
kernel_size=4,
dilations=(1,2,4,8,16,32,64),
padding='causal',
dropout_rate=0.01,
return_sequences=False,
kernel_initializer='he_normal',
use_batch_norm=False)(i)
o = Dense(1)(o)
m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam',loss="hinge")
```
As the code demonstrates, the shape of the input layer is (None, 6, 70), the same as the input format. The shape of the output layer is also (None, 1). The applied optimizer is Adam optimizer and the loss function is hinge loss function.
