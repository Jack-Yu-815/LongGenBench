# LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs
<p align="center">
  <img src="Fig/SGT.jpg" width="500" height="500">
</p>

本仓库提供了论文 **"LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs"**（[https://arxiv.org/abs/2408.07055](https://arxiv.org/abs/2408.07055)）的代码和数据。

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/mozhu/LongGenBench" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/abs/2409.02076" target="_blank">论文</a> 
</p>

### Language/语言: [English](./README.md) | [中文](./readme_zh.md)



## 🔥 更新
**[2025/01/23]** 我们的论文已被ICLR 2025的主会场接收！

## 概述
**LongGenBench** 基准测试旨在评估语言模型（LM）在需要连贯长文本输出的任务中生成长格式内容的能力。传统的基准测试通常关注短上下文任务或特定的信息检索，例如“大海捞针”测试（Needle-in-a-Haystack，NIAH）。与之不同，LongGenBench的设计旨在考察LM生成扩展文本序列的能力，这些序列既要连贯、又要具有丰富的上下文，同时遵循涉及多种约束条件的详细提示指令。

## 基准测试设计
![SGT Benchmark Overview](Fig/SGT_overview.png)

该基准测试评估了10+个长上下文语言模型（LM），通过一系列四种场景和多个子场景进行测试，每个子场景根据提示指令的类型（单实例、范围、周期性）有所不同。这些场景模拟了如城市规划、日记条目或菜单规划等现实世界任务，其中语言模型必须将特定事件、细节或约束条件融入到长格式文本序列中。

## 评估重点
LongGenBench特别强调模型在长文本生成任务中遵循复杂指令的能力，挑战模型在16K和32K标记长度的提示下进行生成。此设置测试模型在长时间段内保持连贯性和相关性的能力，这是自动内容创作、学术摘要和叙事生成等领域应用中的关键衡量标准。

## 使用方法

### Clone this repository:

```bash
git clone git@github.com:mozhu621/LongGenBench.git
cd SGT
pip install -r requirements.txt
```

### Evalution:
```bash
cd ./Evalution
bash Run_short_all_small_model.sh
bash Run_short_all_large_model.sh
bash Run_long_all_small_model.sh
bash Run_long_all_large_model.sh
```

### Static:
```bash
cd ./Evalution/results
Run all cells in sequence --- static.ipynb
```


### Result:
![SGT Benchmark Overview](Fig/result.png)


<!--
### Acknowledge

Due to unforeseen circumstances, the title of my paper shares some similarities with the EMNLP 2024 Findings paper titled *LongGenBench*. However, the focus of our paper is different.

The paper explores model output length under long input conditions by concatenating GSM8K and MMLU data. 

For more details, you can read their paper here: [EMNLP 2024 Findings: LongGenBench: Long-context Generation Benchmark](https://aclanthology.org/2024.findings-emnlp.48/).  
-->

## Citation

如果您觉得这项工作对您的研究有用，请引用我们的论文:

```bibtex
@misc{wu2024longgenbenchbenchmarkinglongformgeneration,
      title={LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs}, 
      author={Yuhao Wu and Ming Shan Hee and Zhiqing Hu and Roy Ka-Wei Lee},
      year={2024},
      eprint={2409.02076},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.02076}, 
}
