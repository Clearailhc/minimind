import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    预训练数据集类，用于处理和准备预训练数据
    每个样本独立处理，不进行样本拼接(packing)
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集
        
        参数:
            data_path: 数据文件路径
            tokenizer: 分词器实例
            max_length: 序列最大长度，默认512
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        从JSONL文件加载数据
        
        参数:
            path: 数据文件路径
            
        返回:
            样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """返回数据集中样本数量"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取指定索引的样本
        注意：每个样本独立处理，不进行样本拼接
        
        参数:
            index: 样本索引
            
        返回:
            处理后的样本，包含输入序列、目标序列和损失掩码
        """
        sample = self.samples[index]

        # 构建输入文本，添加起始和结束标记
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        
        # 使用tokenizer处理文本
        # max_length: 限制序列最大长度
        # padding='max_length': 将所有样本填充到相同的最大长度
        # truncation=True: 如果样本超过最大长度则截断
        # 注意：这里不进行样本拼接，每个样本独立填充到max_length
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',  # 填充到最大长度
            truncation=True,       # 超长则截断
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        
        # 创建损失掩码，用于在计算损失时忽略填充部分
        # 对于实际内容，掩码值为1；对于填充部分，掩码值为0
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 准备自回归训练数据
        # X是输入序列(从0到n-1)，Y是目标序列(从1到n)
        # 例如，对于序列[A,B,C,D]：X=[A,B,C]，Y=[B,C,D]
        # 模型根据当前token预测下一个token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入序列，去掉最后一个token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)   # 目标序列，去掉第一个token
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 损失掩码，与目标序列对齐
        
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    监督微调(SFT)数据集类，用于处理对话数据
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化SFT数据集
        
        参数:
            jsonl_path: 数据文件路径
            tokenizer: 分词器实例
            max_length: 序列最大长度，默认1024
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 获取助手回复的开始和结束标记的token ID
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        """返回数据集中样本数量"""
        return len(self.samples)

    def load_data(self, path):
        """
        从JSONL文件加载对话数据
        
        参数:
            path: 数据文件路径
            
        返回:
            样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        构建符合ChatML格式的对话提示
        
        参数:
            conversations: 对话内容列表
            
        返回:
            格式化后的对话文本
        """
        messages = []
        for i, turn in enumerate(conversations):
            # 根据索引确定角色（偶数为用户，奇数为助手）
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        # 应用聊天模板，但不进行分词和添加生成提示
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        """
        生成损失掩码，只对助手回复部分计算损失
        
        参数:
            input_ids: 输入token ID序列
            
        返回:
            损失掩码，助手回复部分为1，其他部分为0
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 查找助手回复的开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 查找助手回复的结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 对助手回复部分设置掩码为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        参数:
            index: 样本索引
            
        返回:
            处理后的样本，包含输入序列、目标序列和损失掩码
        """
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # 对提示进行分词，并限制长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 填充到最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码，只对助手回复部分计算损失
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入序列
        Y = torch.tensor(input_ids[1:], dtype=torch.long)   # 目标序列
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置的损失掩码

        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    直接偏好优化(DPO)数据集类，用于处理偏好对比数据
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化DPO数据集
        
        参数:
            file_path: 数据文件路径
            tokenizer: 分词器实例
            max_length: 序列最大长度，默认4096
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 获取助手回复的开始和结束标记的token ID
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        """返回数据集中样本数量"""
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        参数:
            index: 样本索引
            
        返回:
            处理后的样本，包含首选和拒绝的回复对
        """
        item = self.data[index]
        # 获取首选和拒绝的回复
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        
        # 应用聊天模板
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # 对首选回复进行编码和填充
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        # 对拒绝回复进行编码和填充
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 获取token ID序列
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        # 准备训练数据
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        """
        生成损失掩码，只对助手回复部分计算损失
        
        参数:
            input_ids: 输入token ID序列
            
        返回:
            损失掩码，助手回复部分为1，其他部分为0
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 查找助手回复的开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 查找助手回复的结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 对助手回复部分设置掩码为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


if __name__ == "__main__":
    pass
