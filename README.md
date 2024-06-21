## Seq2Seq Beam Search Decoding for Pointer Generator 

This code is improved from [jojonki's repository](https://github.com/jojonki/BeamSearch/tree/master) 

The idea of his approach is nice which uses Node and Linked List structure. 

But the implementation of Beam Search seems not true: 
- About definition of a step 
- And finding end nodes is not correct 

![Blank diagram.png](..%2F..%2F..%2FDownloads%2FBlank%20diagram.png)

I fixed a little bit in his code. And the performance of my Pointer Generator is increased from 3 to 5 points bleu score.

My code can be modified to use in any RNN-based Seq2Seq architecture 