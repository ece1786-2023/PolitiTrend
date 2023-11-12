from googletrans import Translator

# 创建一个Translator对象
translator = Translator()

# 德语文本
german_text = "Das ist ein einfacher Satz."

# 翻译成英文
translated_text = translator.translate(german_text, src='de', dest='en')

# 输出翻译结果
print(translated_text.text)