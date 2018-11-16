# RossBot
Work in progress 
HTML Friends scripts from: https://fangj.github.io/friends/

html2text to make scripts txt files
https://github.com/Alir3z4/html2text

Research: 
Models from: https://awesome-chatbot.readthedocs.io/en/latest/README/

https://chatterbot.readthedocs.io/en/stable/index.html
- Can easily train but lots of parameters are abstracted away
- Uses search and classification, NOT a seq2seq model 

https://github.com/Conchylicultor/DeepQA
- Attempt at replicating the results from Google Brain's "Neural Conversation Model" (https://arxiv.org/pdf/1506.05869.pdf)
- Basic implementation that uses an 2 LSTM layers to implement an encoder/decoder
- Use sampled softmax loss function  

http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks
- LSTMs for language translation, English -> French

https://github.com/mckinziebrandon/DeepChatModels
- Examine DynamicBot more closely
 
https://arxiv.org/abs/1412.2007
- Section 3 describes sampled softmax

https://arxiv.org/abs/1406.1078
- RNN Attention paper

https://arxiv.org/pdf/1508.04025.pdf
- Practical RNN attention implementations 

https://www.bioinf.jku.at/publications/older/2604.pdf
- Creation of LSTMs

https://arxiv.org/abs/1611.02344
- Convolution and LSTMs for Machine Translation

https://arxiv.org/abs/1705.03122
- Convolutional with LSTMs implementation

http://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting
- Convolutional LSTMs for precipitation forecasting

https://arxiv.org/pdf/1708.00818.pdf
- Startrek-like bot 
- Weird branching, Startrek vs normal conversation

https://www.quora.com/How-do-you-design-the-personality-of-a-chatbot
- Oftentimes add personality to chatbot to make customer experience more relatable/human
- Try to make conversation more fluid and natural, not try to emulate a target personality 

Stanford CS20Si very useful
- http://web.stanford.edu/class/cs20si/lectures/slides_13.pdf
       -> About chatbots using RNN
- http://web.stanford.edu/class/cs20si/lectures/slides_11.pdf
       -> RNN history/background

https://arxiv.org/pdf/1412.3555v1.pdf
- LSTM vs GRU, about equal performance 

http://www.aclweb.org/anthology/C16-1229
- Convolution and RNN combination

Done:
Cleaned all in clean_scripts

TODO:
apply pairing script to everything
