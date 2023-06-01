# NBCEonChatGLM6b
解决原作者基于LLaMA的NBCE(Naive Bayes-based Context Extension)代码不适配ChatGLM-6b的问题。
使用朴素贝叶斯思想来扩展LLM的Context处理长度，用batch读文本数据，文本生成基于batch内投票。目前没有文本片段分割。
效果受限于LLM，未预训练的6b容易出错

版本环境不同，可能会导致inputs.attention_mask维度不同，如果是2维，运行main.py，如果是4维运行test.py  

15G显存在input_ids维度为[3,270]开始不够了
# 参考
https://github.com/bojone/NBCE
