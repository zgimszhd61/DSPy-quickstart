
# tutorial : https://www.youtube.com/watch?v=Jfpxjg8xj9w
import dspy

class shakeT(dspy.Signature):
    simple_english = dspy.InputField()
    shakes_english = dspy.OutputField()

# 定义一个问题
question = "What are the benefits of using a high-frequency trading system in forex?"

# 使用dspy分析问题并生成答案
turbo = dspy.OpenAI(model='gpt-3.5-turbo',api_key="sk-proj-")
dspy.settings.configure(lm=turbo)

from dspy.signatures.signature import signature_to_template

shakes = signature_to_template(shakeT)

print(str(shakes))

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(shakeT)
    
    def forward(self,simple_english):
        return self.prog(simple_english=simple_english)
    
c = CoT()
ans = c.forward("hello , you should relax and have fun while it lasts.")
print(ans)
