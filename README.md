## Seq2Seq Beam Search Decoding for Pointer Generator 

This code is improved from [jojonki's repository](https://github.com/jojonki/BeamSearch/tree/master) 

The idea of his approach is nice which uses Node and Linked List structure. 

But the implementation of Beam Search seems not correct: 
- About definition of a step 
- And finding end nodes is not correct
  
<div align="center">
  <img src="https://github.com/dinhngoc267/beam_search_seq2seq/assets/49720223/5d6f778d-dad7-4916-a64c-308cdebd6728" width="400">
</div>

I fixed a little bit in his code. And the performance of my Pointer Generator is increased from 3 to 5 points bleu score.

My code can be modified to use in any RNN-based Seq2Seq architecture 
