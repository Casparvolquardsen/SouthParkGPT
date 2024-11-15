# SouthparkGPT: A Small Language Model for Generating South Park Episodes

### Authors: Kerim Erekmen, Karim Djemai, and Caspar Volquardsen

---

## Abstract

This report details the creation of a data-efficient language model, specifically designed to generate script content for the television show *South Park*. The project focuses on constructing a decoder-only transformer-based language model trained from scratch using a dataset of *South Park* episode scripts, with additional pre-training on a subset of the Gutenberg corpus. The research, conducted as part of a master's project at the University of Hamburg, aimed to develop a model that could generate text in the distinctive style of *South Park* while operating within the constraints of limited data and computational resources. The findings indicate that the model, particularly when pre-trained, effectively captures the linguistic style of *South Park* and can generate coherent, contextually appropriate dialogue. The report concludes with a discussion on the challenges encountered and potential areas for future improvement, such as exploring alternative pre-training datasets and different fine-tuning methods. The project represents an initial initiative for sample-efficient text creation, particularly for the creation of media content.

---

## 1. Introduction

### Background
The use of large language models (LLMs) for script generation and storytelling has gained significant attention in recent years, especially within the domains of creative writing and entertainment. LLMs, powered by transformer architectures, have demonstrated remarkable capabilities in generating coherent and contextually relevant text, making them valuable tools for automating aspects of content creation. However, the deployment of such models often demands extensive computational resources and large datasets, which can be a barrier in certain applications.

This report explores the development and implementation of a data-efficient language model, Small Language Model (SLM), tailored to generate script content specifically for the television show *South Park*. The model is designed to produce high-quality text with minimal computational and data requirements, aligning with broader goals of efficiency in natural language generation.

At the core of this project is the construction of a custom decoder-only transformer-based language model, trained from scratch using a dataset composed primarily of *South Park* episode scripts, supplemented by additional English text from the Gutenberg corpus for pre-training.

### Objectives
This research was conducted as part of the master's project "Efficient Methods for Machine Learning" at the University of Hamburg, under the supervision of Sören Laue and Michaela Regneri. The primary objective was to develop a language model capable of efficiently generating script content in the distinctive style of *South Park* while operating within the constraints of limited data and computational resources.

---

## 2. Data

### 2.1 South Park Scripts
#### Source and Extraction
- **Source:** Scripts from all *South Park* episodes, sourced from *South Park Archives*.
- **Extraction:** A Python-based web crawler (using BeautifulSoup) parsed the site’s content, isolating and extracting the relevant HTML table elements for each episode. Extracted tables were stored as CSV files, each representing a single episode.

#### Dataset Format
- **Rows:**
  - Scene descriptions: The first column is empty, and the second column provides scene details.
  - Dialogue: The first column lists character names, and the second column contains their spoken lines.
- **Metadata:**
  - Headers and footers: Each file begins with the episode name and ends with a line indicating the end of the episode.

#### Preprocessing Challenges
1. **Special Characters:** Non-English text and subtitles, such as Chinese and Cyrillic characters.
2. **Formatting Issues:** Errors like missing spaces and occasional misspellings.
3. **Metadata Noise:** Presence of episode titles and markers in the raw dataset.

#### Cleaning
- Removed special characters and metadata lines.
- Focused on simplifying the dataset while retaining all essential script content.

---

### 2.2 Pre-Training Dataset
- **Source:** Gutenberg corpus (10GB, ~48,000 English books) from Hugging Face datasets.
- **Alignment:** Selected to complement the *South Park* scripts by pre-training the model on diverse English-language text.
- **Dataset Complexity:** Differences in structure—Gutenberg is prose-heavy, while *South Park* scripts are formatted as dialogues.

---

### 2.3 Tokenization
#### Approach
- Custom Byte Pair Encoding (BPE) tokenizers trained specifically for the *South Park* dataset.
- Used Hugging Face’s `tokenizers` library.

#### Variants
1. **Standard BPE Tokenizers:** Encoded punctuation and whitespaces differently.
2. **Metaspace Tokenizers:** Replaced whitespaces with special tokens for consistency.
3. **GPT-2 Tokenizer:** Used as a baseline for comparison.

#### Evaluation
- Custom tokenizers performed significantly better than the GPT-2 tokenizer in terms of training efficiency and memory usage.

---

## 3. Model and Training

### 3.1 Decoder-Only Transformer
#### Background
The transformer architecture, introduced by Vaswani et al. in 2017 (*Attention Is All You Need*), has become the dominant architecture for LLMs. For this project, a decoder-only transformer was chosen, inspired by models like GPT-2 and GPT-3.

#### Architecture
- The model uses a stack of decoder layers with self-attention and feedforward components.
- Focuses on autoregressive text generation, predicting the next token in a sequence.

---

### 3.2 Experimental Setup
#### Pre-Training
- **Dataset:** Gutenberg corpus.
- **Goal:** Teach the model grammatical structures and common sense knowledge.

#### Fine-Tuning
- **Dataset:** *South Park* scripts.
- **Goal:** Adapt the model to the distinctive style and structure of the *South Park* dialogues.

#### Hyperparameters
- **Optimization:** AdamW optimizer with cosine learning rate scheduler.
- **Regularization:** Dropout rates of 0.0–0.2 to prevent overfitting.

---

### 3.3 Evaluation Metrics
#### Quantitative Metrics
- **Cross-Entropy Loss:** Measures how well the predicted token distribution matches the true distribution.
- **Accuracy:** Assesses correct token predictions.
- **BLEU Score:** Measures text similarity, used unconventionally to monitor overfitting.

#### Qualitative Analysis
- Coherence, dialogue flow, humor, and style consistency were evaluated manually.

---

## 4. Results and Discussion

### 4.1 Tokenization
- Custom BPE tokenizers reduced training time and memory usage significantly compared to the GPT-2 tokenizer.
- No loss in text quality was observed, making BPE tokenizers more efficient for this dataset.

---

### 4.2 Regularization
- Models without dropout overfitted, while those with dropout (0.2) generalized better.

---

### 4.3 Pre-Training
- Pre-trained models converged faster during fine-tuning and generated higher-quality text compared to models trained from scratch.

---

### 4.4 Challenges
- **Small Dataset Size:** Limited data (~10MB) made it challenging to train larger models.
- **Diversity in Scripts:** Each *South Park* episode covers different themes, requiring contextual knowledge.

---

## 5. Conclusion

The development of a data-efficient language model tailored for *South Park* script generation demonstrates the viability of using small, task-specific datasets. By leveraging pre-training and fine-tuning, the project achieved high-quality text generation despite limited resources. Future work could explore:
1. Larger pre-training datasets more closely aligned with *South Park*.
2. Scaling the model size to improve long-context generation.
3. Integrating additional evaluation metrics for style and humor.

---

## References

1. **Leunen, M.-C. van.**  
   *A Handbook for Scholars.*  
   Oxford University Press, 1992.

2. **Taylor, B. N.**  
   *Guide for the Use of the International System of Units (SI).*  
   NIST Special Publication 811, 1995.  
   [Online](http://physics.nist.gov/Document/sp811.pdf)

3. **Hochreiter, S., & Schmidhuber, J.**  
   *Long Short-Term Memory.*  
   Neural Computation, 9(8), 1735-1780. MIT Press, 1997.

4. **Vaswani, A., et al.**  
   *Attention Is All You Need.*  
   CoRR, abs/1706.03762, 2017.  
   [arXiv](http://arxiv.org/abs/1706.03762)

5. **Liu, P. J., et al.**  
   *Generating Wikipedia by Summarizing Long Sequences.*  
   CoRR, abs/1801.10198, 2018.  
   [arXiv](http://arxiv.org/abs/1801.10198)

6. **Bahdanau, D., Cho, K., & Bengio, Y.**  
   *Neural Machine Translation by Jointly Learning to Align and Translate.*  
   arXiv preprint, arXiv:1409.0473, 2014.

7. **Lin, T., et al.**  
   *A Survey of Transformers.*  
   AI Open, 3, 111-132, 2022.  
   [DOI](https://doi.org/10.1016/j.aiopen.2022.10.001)

8. **Loshchilov, I., & Hutter, F.**  
   *SGDR: Stochastic Gradient Descent with Warm Restarts.*  
   arXiv preprint, arXiv:1608.03983, 2016.

9. **Weights & Biases.**  
   *Weights & Biases.*  
   [Website](https://www.wandb.com)

10. **South Park Archives.**  
    *South Park Archives.*  
    [Website](https://southpark.fandom.com/wiki/South_Park_Archives) (Accessed: August 15, 2024)

11. **Hugging Face.**  
    *English Gutenberg Corpus Dataset.*  
    [Website](https://huggingface.co/datasets/sedthh/gutenberg_english) (Accessed: August 15, 2024)

12. **Kirkpatrick, J., et al.**  
    *Overcoming Catastrophic Forgetting in Neural Networks.*  
    Proceedings of the National Academy of Sciences, 114(13), 3521-3526, 2017.

13. **Papineni, K., et al.**  
    *BLEU: A Method for Automatic Evaluation of Machine Translation.*  
    Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, 311-318, 2002.  
    [DOI](https://doi.org/10.3115/1073083.1073135)

14. **Parisi, G. I., et al.**  
    *Continual Lifelong Learning with Neural Networks: A Review.*  
    Neural Networks, 113, 54-71, 2019.  

15. **Radford, A., et al.**  
    *Language Models Are Unsupervised Multitask Learners.*  
    OpenAI Blog, 1(8), 2019.

16. **Brown, T. B., et al.**  
    *Language Models Are Few-Shot Learners.*  
    arXiv preprint, arXiv:2005.14165, 2020.  
    [arXiv](https://arxiv.org/abs/2005.14165)

17. **Zimerman, I., & Wolf, L.**  
    *On the Long Range Abilities of Transformers.*  
    arXiv preprint, arXiv:2311.16620, 2023.

18. **Lin, Y.-T., & Chen, Y.-N.**  
    *LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models.*  
    arXiv preprint, arXiv:2305.13711, 2023.  
    [arXiv](https://arxiv.org/abs/2305.13711)

19. **Stureborg, R., Alikaniotis, D., & Suhara, Y.**  
    *Large Language Models Are Inconsistent and Biased Evaluators.*  
    arXiv preprint, arXiv:2405.01724, 2024.  
    [arXiv](https://arxiv.org/abs/2405.01724)

20. **Karpathy, A.**  
    *GitHub Repository nanoGPT.*  
    [GitHub](https://github.com/karpathy/nanoGPT) (Accessed: August 20, 2024)

21. **Xie, Z., et al.**  
    *The Next Chapter: A Study of Large Language Models in Storytelling.*  
    arXiv preprint, arXiv:2301.09790, 202


