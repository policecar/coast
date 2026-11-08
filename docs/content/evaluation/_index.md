+++
title = "Evaluation"
weight = 65
+++

# Evaluation

Evaluation measures how good a trained model's representations are: what the codes keep of their input, how well they survive corruption, and how one code state compares with another. It is not testing. The [test suite](../testing/) pins the code's current behaviour so that a change cannot alter it unnoticed; it does not judge whether that behaviour is good. Evaluation does judge it, against references, and its numbers are expected to move when the model changes.

- [Frozen Evaluation](frozen-evaluation/): milestone 1 of issue [#82](https://github.com/cimcai/coast/issues/82). It trains a model, freezes it, and measures the codes it gives for held-out and corrupted images.
- [Hyperparameter Sweep](sweep/): many settings of one template, each trained, frozen and scored on tuning images held out of the train set, to choose which settings get a full frozen evaluation.
