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
model_path = '/opt/wmh_FileFolder/chatglm/chatglm_model'

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
"""

# 拼接context和question
contexts = [''] + contexts  # 添加空Context（无Context预测）
batch = ['User: %s\n\n%s\n\nAssistant:' % (context[:100], question) for context in contexts]
attention_partten = 'bi'
batch = batch[:4]
print('Context长度分布：', [len(text) for text in batch])
print('Context总长度：', sum([len(text) for text in batch]))
print(batch)

def attention_mask_unsqueeze(attention_mask, partten='bi'): # attention_mask_unsqueeze

    SHAPE = (len(attention_mask), 1, len(attention_mask[0]), len(attention_mask[0]))
    _attention_mask = torch.zeros(SHAPE)

    if partten == 'bi':
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                #print(attention_mask[i][j].shape)
                if attention_mask[i][j] == 0:
                    _attention_mask[i, 0, :, j] = 0
                else:
                    _attention_mask[i, 0, :, j] = 1
    else:
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                #print(attention_mask[i][j].shape)
                if attention_mask[i][j] == 0:
                    _attention_mask[i, 0, :, j] = 0
                else:
                    _attention_mask[i, 0, j:, j] = 1
    #print('_att')
    #print(_attention_mask.shape)
    return _attention_mask.bool().cuda()

def attention_mask_squeeze(_attention_mask,partten='bi'): # attention_mask_squeeze
    attention_mask = torch.sum(_attention_mask,dim=2).reshape(_attention_mask.shape[0],-1)
    if partten=='bi':
        #print(attention_mask)
        return ((attention_mask==attention_mask.shape[-1])==True).long()
    else:
        return ((attention_mask!=0)==True).long()

def partten_clf(attention_mask):
    if len(attention_mask.shape) ==2:
        return 'bi'
    else:
        tmp = torch.sum(attention_mask,dim=2).reshape(attention_mask.shape[0],-1)
        #print(tmp)
        if abs(tmp[-1,-1] - tmp[-1,-1-1]) ==1:
            return 'unbi'
        else:
            return 'bi'
@torch.inference_mode()
def generate(max_tokens):
    """Naive Bayes-based Context Extension 演示代码
    """

    inputs = tokenizer(batch, padding='longest', return_tensors='pt', return_attention_mask=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    #attention_mask = attention_mask_unsqueeze(attention_mask)
    #print('input attention_mask shape', attention_mask.shape)

    n = input_ids.shape[0]
    attention_partten = partten_clf(attention_mask)

    res = ''
    for i in range(max_tokens):
        # 模型输出
        #print(f'第{i+1}token开始输出')
        if len(attention_mask.shape) == 2:
            _attention_mask = attention_mask_unsqueeze(attention_mask, attention_partten).cuda()  #四维
        else:
            _attention_mask = attention_mask    #四维
            attention_mask = attention_mask_squeeze(_attention_mask, attention_partten)  #两维

        #print('cal attention_mask shape',_attention_mask.shape)
        outputs = model(input_ids=input_ids,
                        attention_mask=_attention_mask==False,
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
        print(s)
        res += s[0]
        if s[0] == '<eop>':
            break

        # prepare for next iteration
        input_ids = torch.cat([input_ids, next_tokens.tile(n, 1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)
        #print('end attention_mask shape',attention_mask.shape)
    print(question)
    print(re.split('<n><n>',re.sub('▁|<eop>','',res)))
    #['据公开报道,截至2021年6月,吉利德公司有约16,000名员工。', '领英计划裁员716人。', 'Pharmasset被吉利德以110亿美元收购。']
if __name__ == '__main__':
    generate(1000)
