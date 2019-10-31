# NeuRec
## An open source neural recommender library

**Main Contributors**: [BinWu](https://github.com/wubinzzu), [ZhongchuanSun](https://github.com/ZhongchuanSun), [XiangnanHe](https://github.com/hexiangnan), [XiangWang](https://xiangwang1223.github.io), & [Jonathan Staniforth](https://github.com/jonathanstaniforth)

NeuRec is a comprehensive and flexible Python library for recommender systems that includes a large range of state-of-the-art neural recommender models. This library aims to solve general, social and sequential (i.e. next-item) recommendation tasks, using the [Tensorflow](https://www.tensorflow.org/) library to provide 31 models out of the box. NeuRec is [open source](https://opensource.org) and available under the [MIT license](https://opensource.org/licenses/MIT).

## Features

* **Cross platform** - run on any operating system with the available Docker images;
* **State-of-the-art** - 26 neural recommender models available out of the box;
* **Flexible configuration** - easily change the configuration settings to your exact requirements;
* **Easy expansion** - quickly include models or datasets into NeuRec;
* **Fast execution** - naturally supports GPU, with a mutli-thread evaluator;
* **Detailed documentation** - extensive documentation available as Jupyter notebooks at /docs.

## Architecture

The architecture of NeuRec is shown in the diagram below:

<img src="architecture.jpg" width = "50%" height = "50%"/>

## Quick start

Start using NeuRec in three steps with pip.

Firstly, create a properties files to configure NeuRec, including all the necessary settings inside this file:

```bash
vim neurec.properties
```

Secondly, install NeuRec using pip:

```bash
pip3 install neurec
```

Finally, import NeuRec and run as follows:

```python
import neurec

neurec.setup('neurec.properties')
neurec.run()
```

> Additional installation methods, including via Docker, are available at **/docs/1. Installation.ipynb**

## Models

The list of available models in NeuRec, along with their paper citations, are shown below:

| General Recommender | Paper                                                                                                                           |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------|
| GMF, MLP, NeuMF     | Xiangnan He et al., Neural Collaborative Filtering , WWW 2017.                                                                  |
| BPRMF               | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.                                     |
| SBPR                | Tong Zhao et al., Leveraging Social Connections to Improve Personalized Ranking for Collaborative Filtering. CIKM 2014.         |
| FISM                | Santosh Kabbur et al., FISM: Factored Item Similarity Models for Top-N Recommender Systems. KDD 2013.                           |
| NAIS                | Xiangnan He et al., NAIS: Neural Attentive Item Similarity Model for Recommendation . TKDE2018.                                 |
| DeepICF             | Feng Xue et al., Deep Item-based Collaborative Filtering for Top-N Recommendation. TOIS 2019.                                   |
| ConvNCF             | Xiangnan He et al., Outer Product-based Neural Collaborative Filtering . IJCAI 2018.                                            |
| DMF                 | Hong-Jian Xue et al., Deep Matrix Factorization Models for Recommender Systems. IJCAI 2017.                                     |
| CDAE, DAE           | Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.                                  |
| MultiDAE, MultiVAE  | Dawen Liang, et al., Variational autoencoders for collaborative filtering. WWW 2018.                                            |
| JCA                 | Ziwei Zhu, et al., Improving Top-K Recommendation via Joint Collaborative Autoencoders. WWW 2019.                               |
| IRGAN               | Jun Wang, et al., IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models. SIGIR 2017.    |
| CFGAN               | Dong-Kyu Chae, et al., CFGAN: A Generic Collaborative Filtering  Framework based on Generative Adversarial Networks. CIKM 2018. |
| APR                 | Xiangnan He, et al., Adversarial Personalized Ranking for Recommendation. SIGIR 2018.                                           |
| SpectralCF          | Lei Zheng, et al., Spectral Collaborative Filtering. RecSys 2018.                                                               |
| NGCF                | Xiang Wang, et al., Neural Graph Collaborative Filtering. SIGIR 2019.                                                           |
| WRMF                | Yifan Hu, et al., Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.                                            |

| Sequential Recommender | Paper                                                                                                      |
|------------------------|------------------------------------------------------------------------------------------------------------|
| FPMC, FPMCplus         | Steffen Rendle et al., Factorizing Personalized Markov Chains for Next-Basket Recommendation, WWW 2010.    |
| HRM                    | Pengfei Wang et al., Learning Hierarchical Representation Model for NextBasket Recommendation, SIGIR 2015. |
| NPE                    | ThaiBinh Nguyen et al., NPE: Neural Personalized Embedding for Collaborative Filtering, ijcai 2018.        |
| TransRec               | Ruining He et al., Translation-based Recommendation, SIGIR 2015.                                           |

## Contributions

Please let us know if you experience any issues or have suggestions for new features by submitting an issue under the Issues tab.

## Acknowledgements

The development of NeuRec is supported by the National Natural Science
Foundation of China under Grant No. 61772475. This project is also supported by the National Research Foundation, Prime Minister’s Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="https://github.com/wubinzzu/NeuRec/blob/master/next.png" width = "297" height = "100" alt="next" align=center />
