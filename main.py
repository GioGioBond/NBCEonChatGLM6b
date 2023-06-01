#! -*- coding: utf-8 -*-
# Naive Bayes-based Context Extension (NBCE)
# 使用朴素贝叶斯增加LLM的Context处理长度
# 链接：https://kexue.fm/archives/9617

import json
import re
import time

import torch
from transformers import AutoTokenizer,AutoModel

# 模型路径 chatglm-6b
model_path = '/opt/chatglm/chatglm_model/'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.unk_token

# 加载chatglm-6b模型
#model = AutoModel.from_pretrained(model_path, device_map='auto')
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
device = torch.device('cuda')

# 加载示例Context
contexts = json.load(open('contexts.json'))

# 示例问题集（一次性问多个问题，NBCE自行根据Context逐一输出答案）
question = """请仔细阅读材料，回答下面问题：
- 创新药新巨头吉利德公司有多少个员工？
- 领英计划裁员多少人？
- 吉利德收购Pharmasset的价格是多少？
"""
#['据公开报道,截至2021年6月,吉利德公司有约16,000名员工。', '领英计划裁员716人。', 'Pharmasset被吉利德以110亿美元收购。']
# 拼接context和question
contexts = [''] + contexts  # 添加空Context（无Context预测）
batch = ['User: %s\n\n%s\n\nAssistant:' % (context, question) for context in contexts]

#batch = batch[:4]
print('Context长度分布：', [len(text) for text in batch])
print('Context总长度：', sum([len(text) for text in batch]))
def get_attention_mask(attention_mask):

    SHAPE = (len(attention_mask), 1, len(attention_mask[0]), len(attention_mask[0]))
    _attention_mask = torch.zeros(SHAPE)

    for i in range(len(attention_mask)):
        for j in range(len(attention_mask[i])):
            if attention_mask[i][j] == 0:
                _attention_mask[i, 0, :, j] = 1
            else:
                _attention_mask[i, 0, j:, j] = 0
    # print(_attention_mask)
    return _attention_mask.bool().cuda()

@torch.inference_mode()
def generate(max_tokens):
    """Naive Bayes-based Context Extension 演示代码
    """

    inputs = tokenizer(batch, padding='longest', return_tensors='pt', return_attention_mask=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    n = input_ids.shape[0]

    res = ''
    for i in range(max_tokens):
        # 模型输出
        #print(f'第{i+1}token开始输出')
        _attention_mask = get_attention_mask(attention_mask)
        #print(input_ids)
        outputs = model(input_ids=input_ids,
                        attention_mask=_attention_mask,
                        return_dict=True,
                        use_cache=True
                       )

        # ===== 核心代码开始 =====
        beta = 0.25

        logits = outputs.logits[:, -1]
        logits -= torch.max(logits,dim=1).values.reshape(logits.shape[0],-1)
        probas = torch.nn.functional.softmax(logits.float(), dim=-1)

        logits = probas.log()
        k = (probas * logits).sum(dim=-1)[1:].argmax() + 1
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits = (1 + beta) * logits_max - beta * logits_uncond
        # ===== 核心代码结束 =====

        # 构建分布，采样
        # tau = 0.01  # tau = 1是标准的随机采样，tau->0则是贪心搜索
        # _probas = torch.nn.functional.softmax(logits[None] / tau , dim=-1)
        # _next_tokens = torch.multinomial(_probas, num_samples=1).squeeze(1)

        probas = torch.nn.functional.softmax(logits[None], dim=-1)
        next_tokens = torch.topk(probas,1).indices

        s = tokenizer.convert_ids_to_tokens(next_tokens)
        #print(s)
        res += s[0]
        if s[0] == '<eop>':
            break

        # prepare for next iteration
        input_ids = torch.cat([input_ids, next_tokens.tile(n, 1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)

    print(question)
    print(re.split('<n><n>',re.sub('▁|<eop>','',res)))
if __name__ == '__main__':
    generate(1000)