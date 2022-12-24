# Tensorflow dev certificate training

Notes from the udemy course by "zero to mastery".

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Deep learning and TensorFlow fundamentals](#deep-learning-and-tensorflow-fundamentals)
  - [What is deep learning](#what-is-deep-learning)
  - [Why use deep learning](#why-use-deep-learning)
    - [What deep learning is good for:](#what-deep-learning-is-good-for)
    - [What deep learning is **NOT** good for:](#what-deep-learning-is-not-good-for)
    - [ML / DL differences](#ml--dl-differences)
    - [Common algorithms in ML and DL](#common-algorithms-in-ml-and-dl)
  - [What are neural networks?](#what-are-neural-networks)
    - [Steps in nn learning:](#steps-in-nn-learning)
    - [The neural network redux](#the-neural-network-redux)
    - [Types of learning](#types-of-learning)
  - [What is DL already used for?](#what-is-dl-already-used-for)
    - [Common usecases](#common-usecases)
    - ["New" usecases and breakthroughs](#new-usecases-and-breakthroughs)
  - [What is and why use TensorFlow?](#what-is-and-why-use-tensorflow)
    - [What is Tensorflow?](#what-is-tensorflow)
    - [Why TensorFlow](#why-tensorflow)
  - [What is a Tensor?](#what-is-a-tensor)
    - [Exkurs: Notes on Video what is a tensor? by Dan Fleisch](#exkurs-notes-on-video-what-is-a-tensor-by-dan-fleisch)
      - [Vectors](#vectors)
      - [Finding vector components / projecting](#finding-vector-components--projecting)
      - [Finally: tensors](#finally-tensors)
  - [Course information](#course-information)
  - [Creating your first tensors](#creating-your-first-tensors)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Deep learning and TensorFlow fundamentals

### What is deep learning

- Machine learning: turning data into numbers and **finding patterns** in those numbers. 
- Deep learning: subset of machine learning based on neural networks and multiple layers of processing to extract progressively higher level features from data

"Traditional" programming vs ML algorithm: 

- Traditional applies rules to get to an output
- ML figures out rules from a given input and existing ideal output (training)

### Why use deep learning

It's hard to impossible to figure out all the rules for a complex problem, e.g. self-driving cars, image recognition

#### What deep learning is good for:

- Problems with long lists of rules
- Continually changing environments where DL can adapt to new scenarios
- Discovering insights within large collections of data that are impossible to hand-craft as rules, e.g object recognition


#### What deep learning is **NOT** good for:

- when explainability is needed (ML patterns can't be interpreted by humans)
- If a **simple rule-based system** can be used to solve a problem, do that instead of using ML (from [rules of ml](https://developers.google.com/machine-learning/guides/rules-of-ml))
- When errors are unacceptable
- When not much data exists about the problem

#### ML / DL differences

- ML works best on **structured data**, e.g. tables with columns (=features)
- DL performs better on unstructured data such as natural language texts or images

#### Common algorithms in ML and DL

ML:

These are also called "shallow algorithms" 

- random forest
- naive bayes
- nearest neighbour
- support vector machine
- ...many more

DL:

- Neural networks
- fully connected neural network
- convolutional neural network
- recurrent neural network
- transformer
- ...many more

### What are neural networks?

Network or circuit of (artificial) neurons or nodes

#### Steps in nn learning:

- turn inputs / data into numbers (numerical encoding); 
- feed the numbers into a neural network for it to learn representation (= find patterns, features, weights in these numbers)
- network creates / derives representation outputs
- Take representation outputs and convert them into human understandable outputs, e.g. "this picture is of spagghetti", "this soundwave contains the sentence "hey siri, what's up""

#### The neural network redux

![neural network image](./readme_images/neural_network.png)

- Data goes into the input layer
- hidden layers / hidden neurons learn patterns in the data
  - other terms for "patterns": embedding, weights, feature representation, feature vectors
- output layer outputs the learned representations or prediction probabilities

#### Types of learning

- supervised learning: learn from existing data and labels (e.g. a list of images and associated labels such as "Ramen", "Steak" etc.)
- semi-supervised learning: only has *some* labels for the data
- unsupervised learning: only data exists; the nn tries to figure out patterns within the data
- transfer learning: use patterns from an existing *model* (= the output of a neural network / machine learning algo) and try to apply it to other problems / set of data, e.g. 

### What is DL already used for?

- literally anything as long as the input can be converted into numbers

#### Common usecases

- recommendation systems (e.g. youtube recommendations)
- translation (**sequence to sequence** = **seq2seq**, sequence of words gets transformed / translated to another sequence)
- speech recognition (also seq2seq)
- computer vision (**classification / regression**)
- natural language processing (NLP), e.g. "is this mail spam?" (classificatoin / regression)

#### "New" usecases and breakthroughs

- Protein folding for medical research
- ChatGPT
- creating images from commands

### What is and why use TensorFlow?

http://tensorflow.org

#### What is Tensorflow?

- End-to-end deep learning platform
- Write fast (gpu/tpu enabled) deep learning code in python or other accessible languages
- access many pre-built DL models via Tensorflow hub
- whole stack: preprocess data, model data, deploy model in appllication
- originally designed for in-house use by google, now open source

#### Why TensorFlow

- easy model building with high-level APIs
- easy to experiment with
 
### What is a Tensor?

A Tensor is the numerical encoding of information (e.g. images, a text etc.), as well as the *output* - the patterns the neural network has learned - of the neural network.

![this is the flow in tensor flow](./readme_images/tensor_flow.png)  
*this is the "flow" in tensor flow*

#### Exkurs: Notes on Video [what is a tensor?](https://www.youtube.com/watch?v=f5liqUk0ZTw) by Dan Fleisch

##### Vectors

- Vectors: an "arrow" having magnitude and direction (length = magnitude, orientation = direction). Vectors can represent velocity, the direction and magnitude of the earths magnetic field, an area (length = square meters, direction = perpendicular to the area) etc.
- Vectors are members of a wider class called Tensors.

In a cartesian coordinate system there exist **basis vectors** (or *unit vectors*) of the length of one unit (whatever unit the vectors are measured in), one for each direction of the coordinate axis, e.g. x, y, z. These are often represented with a "hat", e.g. ẑ for the z unit vector (pronounced z-hat).

![basis vectors](readme_images/unit_vectors.png)

##### Finding vector components / projecting

To find the vector components (e.g. the x/y coordinates of the "tip" of the vector with its root in x:0, y:0 in a 2d system) we project the vector on each of its axis'.

![vector projections](./readme_images/vector_projection.png)

A vector can thus be represented by its vector components, e.g. points on the x and y axis such as `[3,4]` as the vector always starts at 0,0 and the magnitude / length of the vector can be derived from these (`a^2 + b^2 = c^2`). 

##### Finally: tensors

>Scalars and vectors are special cases of tensors.  
>The types of tensors that are easiest to think about are rank-0 (scalars), rank-1 (vectors), and rank-2 (you can loosely think of a rank-2 tensor as a 3x3 matrix in this context).

*(some person on reddit)* 

![rank 2 tensor](./readme_images/rank-2-tensor.png)

### Course information

Topics covered:

- tensorflow basics
- preprocessing data
- building / using deep learning models
- fitting a model to data
- making predictions with a model (*using* the patterns)
- evaluate model predictions
- saving and loading models
- using a trained model to make predictions on custom data (*new* data sets)

Workflow:

![workflow](./readme_images/workflow.png)


How to approach this course:

- write code / follow along
- explore and experiment; figure out what works and what doesn't
- visualize (recreate in a way you can understand)
- ask questions
- do the exercises
- share your work (doing that with this readme)
- don't overthink the process
- Intelligence is a result of knowledge, so there's no "I can't learn it"

### Creating your first tensors 

Site used for coding: https://colab.research.google.com/

Useful commands:

`strg-enter`: run current cell
`alt-enter`: run cell and inster new
`strg-F9`: run all cells
`strg-F8`: run all cells before current
`strg-m b`: open new code cell below
`strg-m `: run current cell
`strg-m d`: delete cell
`strg-m y`: convert to code cell
`strg-m m`: convert to text cell


Notebook with comments in colab_notebooks/00_tensorflow_fundamentals

