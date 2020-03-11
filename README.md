# NLP Adversarial Attack

Course Project for [COMS 6998-10: Robustness and Security in ML Systems](http://www.cs.columbia.edu/~junfeng/20sp-e6998/), Spring 2020 at Columbia University.

Group members:
- Weifan Jiang (wj2301)
- Haoxuan Wu (hw2754)

## Goal

We try to generate adversarial examples for NLP models with black-box access (only predictions and confidence of predictions are accessible). We aim to produce adversarial example with sentence-level perturbation.

## Approach

The main idea is to combine the [Genetic Algorithm approach](https://www.aclweb.org/anthology/D18-1316/) and  [PWWS](https://www.aclweb.org/anthology/P19-1103.pdf).

Consider the input to the model is a paragraph consists of multiple sentence, then we can compute the _saliency score_ of each sentence `s` by removing `s` from the paragraph, and measure the change in prediction confidence of correct label.

Then we introduce a procedure `Perturb`, to give perturbation to a paragraph; `Perturb` will first randomly select a sentence from paragraph with weighted probability respective to the saliency score of each sentence. Then perturbation will be applied to that sentence.

Finally, we define one more procedure: `x3 = Crossover(x1, x2)` returns a new paragraph `x3` by randomly choosing each sentence from input paragraphs `x1` and `x2`. Now, we can define the genetic algorithm:

```
INPUT:
pop = population count
gen = generation count
x = data point
y = correct label

ALGORITHM:
G1 = empty
for 1 <= i <=pop:
    G0.add(Perturb(x))

for 1 <= i <= gen:
    if any x' in Gi causes Predict(x') != y:
        return x' as advtersarial example and terminate
    
    G(i + 1) = empty
    
    for 1 <= j <= pop:
        parent1 = random sample from Gi with weighted probability respective to probability of incorrect prediction
        parent2 = random sample from Gi with weighted probability respective to probability of incorrect prediction
        
        child = Crossover(parent1, parent2)
        child = Perturb(child)
        
        G(i + 1).add(child)
```

The idea is to apply small perturbations that preserves semantic meaning to sentences while letting the prediction slowly converge to misclassification.
