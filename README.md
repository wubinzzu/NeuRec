# NeuRec
Next RecSys Library
[![GitHub version](https://badge.fury.io/gh/wubinzzu%2FNeuRec.svg)](https://badge.fury.io/gh/wubinzzu%2FNeuRec)

**Founder**: [WuBin]( https://github.com/wubinzzu)<br>
**Main Contributors**: [ZhongchuanSun](https://github.com/ZhongchuanSun) [XiangnanHe](https://github.com/hexiangnan) 

**NeuRec** is is a flexible and comprehensive library including a variety of state-of-the-art neural recommender models. It aims to solve general and sequential (ie., next-item ) recommendation task. Current version includes 20+ neural recommendation models, and more will be be expected in the near future.

<h2>Architecture of NeuRec</h2>

![Architecture](architecture.jpg?raw=true "Title") 

<h2>Features</h2>
<ul>
<li><b>Cross-platform</b>: It is written in python and can be easily deployed and executed in multiple platforms, including Windows, Linux and Mac OS.</li>
  
<li><b>Rich State-of-The-Arts</b>: More than 20 neural recommender models have been implemented, eg., DeepMF, NeuMF, ConvNCF, IRGAN,APR, CFGAN, MultiVAE and more representative models will be continuously added in the NeuRec.
  
<li><b>Flexible Configuration</b>: Configs a recommendation model only using a configuration file, including any loss & optimizer.</li>

<li><b>Easy expansion</b>: Well-designed interfaces, automatic differentiation.</li>
<li><b>Fast execution</b>: Naturally support GPU, multi-thread evaluator. </li>
</ul>

<h2>Prerequisites</h2>
Our framework can be compiled on Python 3.6+ environments with the following modules installed:
- [tensorflow](https://www.tensorflow.org/)
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)

These requirements may be satisified with an updated Anaconda environment as well - https://www.anaconda.com/

<h2>How to Run it</h2>
<ul>
<li>1.Configure the **xx.conf** file in the directory named config. (xx is the name of the algorithm you want to run)</li>
<li>2.Run the **main.py** in the project, and then input following the prompt.</li>
</ul>

<h2>Algorithms Implemented</h2>
<div>

 <table class="table table-hover table-bordered">
  <tr>
		<th>General Recommender</th>
		<th>Paper</th>
  </tr>
	<td scope="row">GMF,MLP,NeuMF</td>
    <td>Xiangnan He et al., Neural Collaborative Filtering , WWW 2017.<br>
    </td>
  </tr>
  <tr>
    <td scope="row">BPRMF</td>
    <td>	Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
     </td>
  </tr> 
  <tr>
    <td scope="row">SBPR</td>
    <td>	Tong Zhao et al., Leveraging Social Connections to Improve Personalized Ranking for Collaborative Filtering. CIKM 2014.
     </td>
  </tr> 
  <tr>
    <td scope="row">FISM</td>
    <td>	Santosh Kabbur et al., FISM: Factored Item Similarity Models for Top-N Recommender Systems. KDD 2013.
     </td>
  </tr> 
  <tr>
    <td scope="row">NAIS</td>
    <td>	Xiangnan He et al., NAIS: Neural Attentive Item Similarity Model for Recommendation . TKDE2018.
     </td>
  </tr> 
    <tr>
    <td scope="row">DeepICF</td>
    <td>	Feng Xue et al., Deep Item-based Collaborative Filtering for Top-N Recommendation. TOIS 2019.
     </td>
  </tr> 
    </tr> 
    <tr>
    <td scope="row">ConvNCF</td>
    <td>	Xiangnan He et al., Outer Product-based Neural Collaborative Filtering . IJCAI 2018.
     </td>
  </tr> 
  <tr>
    <td scope="row">DMF</td>
    <td>	Hong-Jian Xue et al., Deep Matrix Factorization Models for Recommender Systems. IJCAI 2017.
     </td>
  </tr> 
  <tr>
    <td scope="row">CDAE,DAE</td>
    <td>	Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.
     </td>
  </tr> 
 <tr>
    <td scope="row">MultiDAE,MultiVAE</td>
    <td>	Dawen Liang, et al., Variational autoencoders for collaborative filtering. WWW 2018.
     </td>
  </tr>	
 <tr>
    <td scope="row">JCA</td>
    <td>	Ziwei Zhu, et al., Improving Top-K Recommendation via Joint
Collaborative Autoencoders. WWW 2019.
     </td>
  </tr>		
 <tr>
    <td scope="row">IRGAN</td>
    <td>	Jun Wang, et al., IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models. SIGIR 2017.
     </td>
  </tr>	
   <tr>
    <td scope="row">CFGAN</td>
    <td>       Dong-Kyu Chae, et al., CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks. CIKM 2018.
     </td>
  </tr>
     <tr>
    <td scope="row">APR</td>
    <td>       Xiangnan He, et al., Adversarial Personalized Ranking for Recommendation. SIGIR 2018.
     </td>
  </tr>	
	
	
  </table>

  </br>
  <table class="table table-hover table-bordered">
  <tr>
		<th>Sequential Recommender</th>
		<th>Paper</th>
   </tr>
  <tr>
	<td scope="row">BPR</td>
    <td>Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">WRMF</td>
    <td>Yifan Hu et al.Collaborative Filtering for Implicit Feedback Datasets, KDD 2009.
     </td>
  </tr>
  </table>
</div>


