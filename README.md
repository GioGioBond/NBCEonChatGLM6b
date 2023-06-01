# NBCEonChatGLM6b
解决原作者基于LLaMA的NBCE(Naive Bayes-based Context Extension)代码不适配ChatGLM-6b的问题。
使用朴素贝叶斯思想来扩展LLM的Context处理长度，用batch读文本数据，文本生成基于batch内投票。

目前没有文本分割
# 参考
https://github.com/bojone/NBCE
