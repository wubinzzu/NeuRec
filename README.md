# NeuRec
[![GitHub version](https://badge.fury.io/gh/wubinzzu%2FNeuRec.svg)](https://badge.fury.io/gh/wubinzzu%2FNeuRec)

**Founder**: [WuBin]( https://github.com/wubinzzu)<br>
**Main Contributors**: [ZhongchuanSun](https://github.com/ZhongchuanSun) [XiangnanHe](https://github.com/hexiangnan) 

**NeuRec** is is a flexible and comprehensive library including a variety of state-of-the-art neural recommender models. It aims to solve general and sequential (ie., next-item ) recommendation task. Current version includes 20+ neural recommendation models, and more will be be expected in the near future. NeuRec is free software (open source software), it can be used and distributed under the terms of the GNU General Public License (GPL).

<h2>Architecture of NeuRec</h2>

![Architecture](architecture.jpg?raw=true "Title") 

<h2>Features</h2>
<ul>
<li><b>Cross Platform</b>: Written in python and easily executed in multiple platforms, including Windows, Linux and Mac OS.</li>
  
<li><b>Rich State-of-The-Arts</b>: More than 20 neural recommender models have been implemented, eg., DeepMF, NeuMF, ConvNCF, IRGAN,APR, CFGAN, MultiVAE and more representative models will be continuously added in the NeuRec.
  
<li><b>Flexible Configuration</b>: Configs a recommendation model only using a configuration file, including any loss & optimizer.</li>

<li><b>Easy Expansion</b>: Well-designed interfaces, automatic differentiation.</li>
<li><b>Fast Execution</b>: Naturally support GPU, multi-thread evaluator. </li>
</ul>

<h2>Prerequisites</h2>
Our framework can be compiled on Python 3.6+ environments with the following modules installed:
<ul>
<li>tensorflow1.12.0+</li>
<li>numpy1.15.4+</li>
<li>scipy1.1.0+</li>
</ul>
These requirements may be satisified with an updated Anaconda environment as well - https://www.anaconda.com/

<h2>How to Run it</h2>
<ul>
<li>Configure the **xx.conf** file in the directory named config. (xx is the name of the algorithm you want to run)</li>
<li>Run the **main.py** in the project, and then input following the prompt.</li>
</ul>

<h2>Models Implemented</h2>
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
	<td scope="row">FPMC,FPMCplus</td>
    <td>    Steffen Rendle et al., Factorizing Personalized Markov Chains
for Next-Basket Recommendation, WWW 2010.<br>
    </td>
  </tr>
    <tr>
    <td scope="row">HRM</td>
    <td>     Pengfei Wang et al., Learning Hierarchical Representation Model for NextBasket Recommendation, SIGIR 2015.
     </td>   
  </tr>
   <tr>
    <tr>
    <td scope="row">HRM</td>
    <td>     Pengfei Wang et al., Learning Hierarchical Representation Model for NextBasket Recommendation, SIGIR 2015.
     </td>   
  </tr>
	<tr>
    <td scope="row">NPE</td>
    <td>     ThaiBinh Nguyen et al., NPE: Neural Personalized Embedding for Collaborative Filtering, ijcai 2018.
     </td>   
  </tr>
    <tr>
    <td scope="row">TransRec</td>
    <td>    Ruining He et al., Translation-based Recommendation, SIGIR 2015.
     </td>   
  </tr>

  </table>
</div>
</div>
<h2>Acknowledgements</h2>
The development of NeuRec was supported by the National Natural Science
Foundation of China under Grant No. 61772475. This work is also supported by
the National Research Foundation, Prime Minister’s Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="https://github.com/wubinzzu/NeuRec/blob/master/next.png" width = "297" height = "100" alt="next" align=center />

