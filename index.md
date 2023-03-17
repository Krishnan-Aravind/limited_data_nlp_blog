---
---

Welcome to My Home Page

{% assign date = '2020-04-13T10:20:00Z' %}

- Original date - {{ date }}
- With timeago filter - {{ date | timeago }}
- hello

## Unlearning Span corrution

We use pretrained T5 [CITE] models for all experiments. This cannot be done directly for NLP taks, since T5 is pretrained on span corruption. Since the model has learned to ignore certain spans of text or has learned incorrect syntactic or grammatical patterns due to span corruption during pretraining, it may make incorrect predictions about the relationship between the premise and hypothesis. One effective method for resolving this is to train the model using a language modelling objective for additional steps. This training approach helps the model to re-learn the correct syntax and grammar rules that were previously distorted due to the corruption of spans.  The authors therefore pretrain T5 for an additional 100k steps to bring out language modelling capabilities.
