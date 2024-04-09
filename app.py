# 导入 DSPy 库
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
import os

# 设置语言模型
turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
dspy.settings.configure(lm=turbo)

# 从 GSM8K 数据集加载数学问题
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

# 定义一个使用 ChainOfThought 模块进行逐步推理的自定义程序
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

# 使用 BootstrapFewShotWithRandomSearch teleprompter 优化程序
from dspy.teleprompt import BootstrapFewShot

# 设置优化器：我们想要“自举”（即自动生成）4-shot 示例
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# 假设 BootstrapFewShot 类接受一个配置字典作为参数，但参数名不是 'config'
# 我们将直接将配置字典传递给类的构造函数
config = {
    'max_bootstrapped_demos': 4,
    'max_labeled_demos': 4
}

# 初始化 BootstrapFewShot 类实例
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)

optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

# 使用开发数据集评估编译（优化）后的 DSPy 程序的性能
from dspy.evaluate import Evaluate

# 设置评估器，可以多次使用
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# 评估我们的 optimized_cot 程序
evaluate(optimized_cot)

question = "一个长方形的长是10厘米，宽是5厘米，求这个长方形的面积。"

# 使用程序来解决这个问题
answer = optimized_cot(question=question)

# 打印答案
print(f"问题: {question}")
print(f"答案: {answer.answer}")