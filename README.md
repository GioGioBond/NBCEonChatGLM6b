# NBCEonChatGLM6b
解决原作者基于LLaMA的NBCE(Naive Bayes-based Context Extension)代码不适配ChatGLM-6b的问题。  
使用朴素贝叶斯思想来扩展LLM的Context处理长度，用batch读文本数据，文本生成基于batch内投票。目前没有文本片段分割。  



目前已知问题：
- 1、效果受限于chatglm6b。伴随question数量变化，可能会出现之前能答对的问题无法回答的现象。当context长度增长到一定程度时，会表现为特殊token与中文混合输出  
- 2、很占显存，小显存单卡不适合。15G显存在input_ids维度为[4,242]开始不够了，经反馈，32G也会经常爆



目前chatglm6b v1.1.0版本 2023-06-02下载， vocab_size是130528， inputs.attention_mask是四维  
下载地址  https://huggingface.co/THUDM/chatglm-6b

2023-06-02  添加缓存清除；代码适配新版chatglm6b，删除旧版代码
# 参考
https://github.com/bojone/NBCE
