![](header.png)
# German Benchmark Datasets
## Translating Popular LLM Benchmarks to German


Inspired by the [HuggingFace Open LLM leaderboard](HuggingFaceH4/open_llm_leaderboard), this project aims to utilize GPT-3.5 to provide German translations for popular LLM benchmarks, enabling researchers and practitioners to evaluate the performance of various language models on tasks using the German language. We follow the HF leaderboard in designing the datasets to be used with the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). By creating these translated benchmarks, we hope to contribute to the advancement of multilingual natural language processing and foster research in the German NLP community.

All translated datasets are made available on [HuggingFace]():
- [ARC-Challenge-DE](https://huggingface.co/datasets/bjoernp/arc_challenge_de)     
- [HellaSwag-DE](https://huggingface.co/datasets/bjoernp/hellaswag_de)
- [MMLU-DE](https://huggingface.co/datasets/bjoernp/mmlu_de)
- [TruthfulQA-DE](https://huggingface.co/datasets/bjoernp/truthful_qa_de)


## Datasets
We are currently providing translations for the four datasets also used in the [HuggingFace Open LLM leaderboard](HuggingFaceH4/open_llm_leaderboard):
- [ARC-Challenge](https://huggingface.co/datasets/ai2_arc): A dataset created by AI2 of AllenAI (Clark *et al.* 2018), consisting of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. We provide a German translation of the test and eval sets from the challenging subset of the dataset: 1172 questions test and 299 questions eval. License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://huggingface.co/datasets/hellaswag): A dataset created by Zellers *et al.* (2019) to evaluate a model's ability to perform commonsense reasoning. The dataset consists of 59,950 partial sentences with multiple completions each. We provide a German translation of the test set and part of the train set: 10,000 and 1000 questions respectively. License: [MIT](https://github.com/rowanz/hellaswag/blob/master/LICENSE)
- [MMLU](https://github.com/hendrycks/test): This dataset (available [here](https://huggingface.co/datasets/cais/mmlu) and [here](https://huggingface.co/datasets/tasksource/mmlu) on HuggingFace), is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. This covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem-solving ability. We provide translations of test and validation sets for all tasks: 6829 total questions. License: [MIT](https://github.com/hendrycks/test/blob/master/LICENSE)
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa): A benchmark (Lin *et al.* 2021) designed to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. We provide translations of the full test set: 817 questions. License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)


These datasets are designed to cover a range of tasks and serve as a foundation for evaluating German language models. As the project evolves, we plan to add more benchmarks to further enhance the variety and utility of the available datasets.

## Usage
Currently, the best way to evaluate a model on these datasets is to use our our clone of the LM Evaluation Harness: https://github.com/bjoernpl/lm-evaluation-harness-de/tree/mmlu_de. Soon, we will also contribute our changes to the original repository.

To evaluate a model on a dataset, follow these steps:
1. Clone the repository and checkout the `mmlu_de` branch:
    ```
    git clone https://github.com/bjoernpl/lm-evaluation-harness-de/
    cd lm-evaluation-harness-de
    git checkout mmlu_de
    ```
2. Install the requirements:
    ```
    pip install -r requirements.txt
    ```
3. Run evaluation on any of the tasks. `MMLU-DE*`, `hellaswag_de`, `arc_challenge_de`, `truthful_qa_de` are the names. Keep in mind, in the LLM leaderboards, the fewshot numbers `5, 10, 25 and 0` are used respectively. The following example runs a [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model on the HellaSwag dataset with a fewshot of 5 on a GPU:
    ```
    python run.py --model=hf-causal --model_args=llama-2-7b --tasks MMLU-DE* --fewshot 5
    python main.py \
    --model hf-causal \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=float16 \
    --tasks hellaswag_de \
    --num_fewshot 10 \
    --device cuda:0
    ```
    For more details see the original LM Evaluation Harness [README](https://github.com/EleutherAI/lm-evaluation-harness)

## Creation Process
We translated each dataset independently, as each required specific considerations. The code to reproduce the translations is available in the `dataset_translation` folder. While a large part of all examples can be successfully translated with clever prompting, manual post-processing was required to fill in the gaps.

## License
This project is licensed under the [Apache 2.0 License](LICENSE).
However, please note that the original datasets being translated may have their own licenses, so make sure to respect and adhere to them when using the translated benchmarks.

We hope that these translated benchmarks empower the German NLP community and contribute to advancements in multilingual language modeling. Happy coding and researching!

## References
Great thanks to the creators of the original datasets for making them available to the public. Please consider citing the original papers if you use the datasets in your research.

- HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers *et al.* 2019)
    ```
    @inproceedings{zellers2019hellaswag,
        title={HellaSwag: Can a Machine Really Finish Your Sentence?},
        author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
        booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
        year={2019}
    }
    ```

- TruthfulQA: Measuring How Models Mimic Human Falsehoods (Lin *et al.* 2021)
    ```
    @misc{lin2021truthfulqa,
        title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
        author={Stephanie Lin and Jacob Hilton and Owain Evans},
        year={2021},
        eprint={2109.07958},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    ```

- Measuring Massive Multitask Language Understanding (Hendrycks *et al.* 2021)
    ```
    @article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }
    ```

- Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge (Clark *et al.* 2018)
    ```
    @article{allenai:arc,
        author    = {Peter Clark  and Isaac Cowhey and Oren Etzioni and Tushar Khot and
                        Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
        title     = {Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
        journal   = {arXiv:1803.05457v1},
        year      = {2018},
    }
    ```
---

*Disclaimer: This repository is not affiliated with or endorsed by the creators of the original benchmarks.*