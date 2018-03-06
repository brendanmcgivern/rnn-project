# Recurrent Neural Networks: time series prediction and text generation

![Sherlock Time](./images/sherlock-time.png)

## Overview

In this project, I built two RNNs that can generate sequences based on input data - with a focus on two applications: With the first, I used real market data in order to predict future Apple stock prices. The second one was trained on Sir Arthur Conan Doyle's classic novel Sherlock Holmes and generates wacky sentences based on this text.

## Problem 1: Perform time series prediction
Using historical apple stock data, I created an RNN model to forecast the price changes 7 days in advance.

## Cutting the time series into sequences
I could not pass my raw data directly into my model, so I began by cutting my time series data into sequences.  I did this by creating a windowing function which essentially runs a sliding window along the input series and constructs a set of associated input/output pairs to regress on.

## Build an RNN regression model.
Because of Keras' ease of use and abstraction layer, I was able to construct the RNN model with only a few lines of code.

- layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
-  layer 2 uses a fully connected module with one unit
-  the 'mean_squared_error' loss was used because we are performing regression here

After training the model, I was able to achieve a testing error rate of `1.40%`.

![Apple Stock](./images/stock-pred.png)

## Problem 2: Create a sequence generator

In this project, I implemented a popular RNN architecture to create an English language sequence generator capable of building semi-coherent English sentences from scratch by building them up character-by-character.  I used a complete version of Sir Arthur Conan Doyle's classic book The Adventures of Sherlock Holmes, which represented a fairly large training corpus.

## Preprocessing a text dataset
I had to do a fair amount of preprocessing on the dataset to clean it up.  Here is the process I went through:

- Convert all text to lowercase letters.
- Strip out `\n` and `\r`
- Used regex to strip out anything other than ASCII lowercase and the following punctuation - ['!', ',', '.', ':', ';', '?']

With all of the preprocessing out of the way, I was left with a corpus containing 573686 total characters and 37 unique characters

## Cutting data into input/output pairs
Before I can train my model I first need a set of input/output pairs to train it on.

Similar to how I implemented my windowing function in project 1, I will do that same thing here, with the addition of a `step_size`.  The added `step_size` allows me to bypass having to slide the window along one character at a time because I can move by a fixed step size  M.

This is typically done with large input texts (like mine which has over 500,000 characters) when sliding the window along one character at a time we would create far too many input/output pairs to be able to reasonably compute with.

## Text Generation - classification

Since character-by-character text generation is a classification problem, I will need to one-hot encode the 37 unique characters.

I started this by forming a dictionary mapping each unique character to a unique integer, and one dictionary to do the reverse mapping.

```python
# map each unique character to unique integer
chars_to_indices = dict((c, i) for i, c in enumerate(chars))

# map each unique integer back to unique character
indices_to_chars = dict((i, c) for i, c in enumerate(chars))
```

I then wrote a function to use the two dictionaries above to one-hot encode the input/output pairs.

## Building the RNN model
With our dataset loaded and the input/output pairs extracted/transformed, we can now begin setting up our RNN for training.

I built a 3 layer RNN model with the following:

- Layer 1 is an LSTM module with 200 hidden units --> input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
- Layer 2 is a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
- Layer 3 is a softmax activation
- I used the categorical_crossentropy loss

## Results 
After training the model for 30 epochs, it was producing the results shown below.  As you can see, the predicted output primarily consists of semi-coherent English sentences built from scratch!

-------------------

input chars = 
 a nature such as his. and yet there was but one woman to him, and that woman was the late irene adl"

predicted chars = 
er go down at the other. and i should be it least that i was all tones, and i ford you see so at tha"

-------------------
