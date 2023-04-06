# Prompt Tuning and Transfer

# Introduction

Parameter-efficient fine tuning has garnered significant attention in the recent years ever since the size of SOTA models (and the inflation rates) started going up exponentially high. Consequently, several approaches have been proposed to preserve model efficiency while training on a mere fraction of the model’s parameters. Some examples include Adapters[1], BitFit [2] and prefix tuning [3]. A major limitation of these approaches is the fact that they sacrifice model performance in the race for parameter efficiency.

In this blog-post, we will look at **prompt tuning**, a clean and extremely efficient parameter efficient fine tuning method proposed in [4]. We will look at the underlying methods and look at possible extensions of the approach.  

## What is it?

Prompt tuning adds additional trainable tokens at the input and fine tunes them to a specific task instead of fine tuning the entire model. Intuitively, it can be seen as adding a continuous task-description embedding that only the model understands at the beginning of an input, and reusing this embedding across all instances of one task. We improve the task-description embedding with fine tuning and leave the model untouched.

<figure><img src="Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/prompt_tuning_diagram.png">
  <figcaption align = "center">Figure 1: Prompt Tuning [5]
  </figcaption>
</figure>

blah blah blah
![                                                                      Figure 1: Prompt Tuning [5]](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/prompt_tuning_diagram.png)

                                                                      Figure 1: Prompt Tuning [5]

Cool right? Add some embeddings to the input in the beginning and the model does the rest because it has learned to recognize the embeddings to do a specific task. Now let’s see how this works.

# On Your Mark…

Before we start with the technical jargon, let’s define some experimental settings for the tasks we want. This is the target space we will work in across this blog.

| Specifications | Description | Example |
| --- | --- | --- |
| Task Type | NLI  (classification / regression) | COLA (is the sentence Grammatical or Ungrammatical?)  |
| Input | Text | The building is than that one |
| Output | Text | Ungrammatical |
| Model  | Sequence to Sequence | T5 |

For the task scenario depicted in Table 1, we start by picking up a pre-trained T5 [6] model from our dear friends at Google. A direct deployment for the input/output pair shown in the Table is not possible, since T5 has been trained on span corruption, which gives it a distorted understanding of grammar. We start by converting it into a causal LM by additionally pre-training  T5 for 100K steps with a causal language modelling objective (i.e., predict the next word). This is shown in Figure 2.

![                                                    Figure 2: Updating T5 to fit NLU tasks](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/span_corrputionlm_ft(1).png)

                                                    Figure 2: Updating T5 to fit NLU tasks

Now that we have a model that can generate the input-output pairs we want, let’s get busy fine tuning. Or not… 

# Get Prompts…

The typical story here is to fine tune the updated T5 model for each of our tasks. But this approach is parameter inefficient, and we need to store one fine tuned model each for individual tasks. In Prompt tuning, we add trainable prompts to the input and fine tune them for each task. The idea is simple: We add $k$ “imaginary” words before the actual input and let the model fine tune these imaginary words into cues it can understand to do a particular task. The model left untouched and can be reused again for another task, by simply changing the prompts.

![            Figure 3: Adding soft prompts to the input](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/span_corrputionlm_ft_prompt(1).png)

            Figure 3: Adding soft prompts to the input

For the sake of simplification, let us assume that the prompts have been random initialized. The next step is to fine tune them with a frozen model. We use the language-modelling adapted T5 that was built earlier. The soft prompts are trained for 30K steps and viola! we have a trained prompt that can be used for inference for a particular task. And we didn’t touch the model at all! 

![                                                  Figure 4: Training Soft Prompts](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/span_corrputionlm_ft_prompt(3).png)

                                                  Figure 4: Training Soft Prompts

<aside>
💡  The continuous prompts are defined and fine tuned in the embedding space, so don’t expect to map them back into a lexical unit after fine tuning.

</aside>

 In [4], the authors compare Prompt Tuning and other parameter-heavy fine tuning approaches using the SuperGLUE [7] benchmark, and make three significant observations:

- Prompt Tuning uses less than 0.01% task-specific parameters for models with over a billion parameters, becoming the most parameter efficient approach on the market.
- Prompt Tuning competes with fine tuning at large model sizes.
    
    ![             Figure 5: Comparing Prompt Tuning with Fine Tuning (Model Tuning)](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/Screenshot_from_2023-01-10_18-37-03.png)
    
                 Figure 5: Comparing Prompt Tuning with Fine Tuning (Model Tuning)
    
- But the performance gap is still significant for small model sizes: Fine tuning seems to provide significantly better numbers in these cases 💩

# SPoT!

The performance drop of prompt tuning at small model sizes affect a large majority of users who cannot afford large models. The authors of [8] introduce SPoT, a prompt “sharing” mechanism that remedies this issue. The core idea of Soft Prompt Transfer (SPoT) is to reuse the prompts trained for a task in initializing prompts for similar tasks. As shown in Figure 6, we start by tuning prompts for a source task that is close to our target task and simply copy the source prompts for the target task . The target prompts are then tuned as usual.

                                                 Figure 6: Prompt Transfer mechanism

![screenshot.png](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/screenshot.png)

Note that SPoT works exactly like Prompt tuning once prompt transfer is done. The training is equally prompt efficient as the original approach, with differences being confined to the initialization strategy.  **Experiments with the SuperGLUE benchmark show that SPoT outperforms Prompt Tuning and competes with Fine Tuning across all model sizes.** 

![                                                      Figure 7: SPoT Performance](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/Screenshot_from_2023-01-15_16-01-37.png)

                                                      Figure 7: SPoT Performance

## PS: Where to SPoT?

SPoT works exceptionally well when we reuse prompts from similar tasks, but the question remains as to how one finds an optimal task to transfer prompts from. It sounds too time consuming to use a validation set to cross verify transferability across exhaustive combinations, so the authors provide a pipeline to select the optimal source task from a hyperplane of possible options. Their idea is to compare tasks in the prompt domain to see if they are close enough to benefit from each other. The mechanism is as follows:

1. Train prompts for the target task and the source tasks with standard initialization, for 100K steps each
2. Project all resulting prompts into a vector space and compute cosine similarities of every source prompt from the target prompt
3. Choose the source prompt that is closes to the target prompt in terms of cosine similarity.
4. Use this source prompt for SPoT by retraining the target prompt with source initialization

![                              Figure 8: Fining the optimal Source Prompt for SPoT](Prompt%20Tuning%20and%20Transfer%20679d90333015413eb8145a4e8fa68419/Task_Selection.drawio(2).png)

                              Figure 8: Fining the optimal Source Prompt for SPoT

They perform a lot of experiments and confirm that this approach works well enough for everyday programmers. This makes it easy to find the optimal source task to implement prompt transfer. 

# For the Sleepy Reader: Conclusion

This blog post took an eagle’s eye view of Prompt Tuning, a new parameter efficient fine tuning method that seems to break the charts with parameter efficiency while maintaining model performance. The basic idea of Prompt Tuning is to add some tunable “words” to the beginning of the input and let the model fine tune these embeddings to fit a specific task. We saw that Prompt tuning beats all current benchmarks in terms of parameter efficiency and competes with full-fledged model tuning at large model sizes. However, results show that Prompt-Tuning under-performs traditional fine tuning when the model size is low.  We follow it up with Prompt Tuning’s cousin SPoT, where we steal tuned prompts from similar tasks. SPoT extends Prompt Tuning’s comparable performances (with fine tuning) to small scale models as well. We also saw a pipeline to select the optimal source for prompt transfer, from a list of candidates.  

# References

1. Houlsby, Neil, et al. "Parameter-efficient transfer learning for NLP." *International Conference on Machine Learning*. PMLR, 2019.
2. Zaken, Elad Ben, Shauli Ravfogel, and Yoav Goldberg. "Bitfit: Simple parameter-efficient fine tuning for transformer-based masked language-models." *arXiv preprint arXiv:2106.10199* (2021).
3. Li, X.L. and Liang, P., 2021. Prefix-tuning: Optimizing continuous prompts for generation. *arXiv preprint arXiv:2101.00190*
4. Lester, B., Al-Rfou, R. and Constant, N., 2021. The power of scale for parameter-efficient prompt tuning. *arXiv preprint arXiv:2104.08691*
5. Figure taken from [https://ai.googleblog.com/2022/02/guiding-frozen-language-models-with.html](https://ai.googleblog.com/2022/02/guiding-frozen-language-models-with.html)
6. Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." *The Journal of Machine Learning Research*  21.1 (2020): 5485-5551.
7. Wang, Alex, et al. "Superglue: A stickier benchmark for general-purpose language understanding systems." *Advances in neural information processing systems* 32 (2019).
8. Vu, Tu, et al. "Spot: Better frozen model adaptation through soft prompt transfer." *arXiv preprint arXiv:2110.07904*  (2021).
