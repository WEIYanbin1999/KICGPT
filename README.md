# KICGPT
The official repository of EMNLP 2023 Findings paper: "KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion"


#### Section 1. Prepare the Datasets
Before executing the code, please download the fb1k-237 and wn18rr datasets first and organize them into the datasets/ directory according to the repository structure.

For convenience, demonstration pools, retriever preliminary results, text-alignment clean mapping, and all support files for link prediction are involved in our provided datasets.
As a result, You can have a quick trial by directly turning to **Section 4**.


#### Section 2. Demonstration Pools
Running commands to generate demonstration pools
~~~
python get_demonstrations.py --dataset fb15k-237
python get_demonstrations.py --dataset wn18rr
~~~
Demonstration pools are under datasets/fb15k-237/demonstration/ and datasets/wn18rr/demonstration/.


#### Section 3. Text self-alignment
Running commands perform text self-alignment
~~~
python text_alignment_query.py --dataset fb15k-237
python text_alignment_query.py --dataset wn18rr
python text_alignment_process.py --dataset fb15k-237
python text_alignment_process.py --dataset wn18rr
~~~
Text self-alignment chatlog and outputs are under datasets/fb15k-237/alignment/ and datasets/wn18rr/alignment/.



#### Section 4. Link Prediction


##### Link Prediction by directly performing candidate re-ranking
~~~
python3 link_prediction.py --dataset fb15k-237 --query tail
python3 link_prediction.py --dataset fb15k-237 --query head
python3 link_prediction.py --dataset wn18rr --query tail
python3 link_prediction.py --dataset wn18rr --query head
~~~
##### Link Prediction by scoring candidates
~~~
python3 link_prediction_scoring.py --dataset wn18rr --query tail
python3 link_prediction_scoring.py --dataset wn18rr --query head
~~~

After running the link_prediction, the chatlog and output results are under the outputs/ directory. Text self-alignment is optional by appending --align_text at the tail of the following link prediction running commands.






#### Cite:
~~~
@inproceedings{wei2023kicgpt,
  title={KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion},
  author={Wei, Yanbin and Huang, Qiushi and Zhang, Yu and Kwok, James},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={8667--8683},
  year={2023}
}
~~~

