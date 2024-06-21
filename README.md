## Seq2Seq Beam Search Decoding for Pointer Generator 

This code is improved from [jojonki's repository](https://github.com/jojonki/BeamSearch/tree/master) 

The idea of his approach is nice which uses Node and Linked List structure. 

But the implementation of Beam Search seems not correct: 
- About definition of a step 
- And finding end nodes is not correct
  
![](https://github.com/dinhngoc267/beam_search_seq2seq/assets/49720223/2ac67e4f-3b19-4ca6-b8a7-0de3a33b9923 =250x250)




I fixed a little bit in his code. And the performance of my Pointer Generator is increased from 3 to 5 points bleu score.

My code can be modified to use in any RNN-based Seq2Seq architecture 
