import gradio as gr
import torchtext
import torch
import matplotlib.pyplot as plt
import args_parameter
import seaborn as sns

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification,GPT2Tokenizer, GPT2ForSequenceClassification
import numpy 
import BaselineModel
import CNN_model
import torch.nn.functional as F
args = args_parameter.args
glove = torchtext.vocab.GloVe(name="6B", dim=100)

# def check(text,plot_type):
    
#     baseline_model = BaselineModel.BaselineModel(glove)
#     baseline_model.load_state_dict(torch.load('/Users/lifeifan/Desktop/ece1786/a2/A2_Gradio/baseline.pt'))


#     # cnn_model = CNN_model.CNNTextClassifier(glove,args.k1,args.n1,args.k2,args.n2,args.freeze_embedding,args.bias)
#     cnn_model = CNN_model.CNNTextClassifier(glove,args)
#     cnn_model.load_state_dict(torch.load('/Users/lifeifan/Desktop/ece1786/a2/A2_Gradio/cnn.pt'))

#     tokens = text.split()
#     token_ints = [glove.stoi.get(tok, len(glove.stoi)-1) for tok in tokens]
#     token_tensor = torch.LongTensor(token_ints).view(-1,1)

#     print(token_tensor.size())

#     baseline_pre = baseline_model(token_tensor)
#     baseline_prob = float(torch.sigmoid(baseline_pre))
    


#     cnn_pre = cnn_model(token_tensor)
#     cnn_prob = float(torch.sigmoid(cnn_pre))


#     # if baseline_prob > 0.5:
#     #     bnn_so,baseline_prob = "Subjective", baseline_prob
#     # else:
#     #     bnn_so,baseline_prob = "Objective", 1-baseline_prob

#     if cnn_prob > 0.5:
#         cnnso,cnn_prob = "progress", cnn_prob
#     else:
#         cnnso,cnn_prob = "conservative", 1-cnn_prob


#     return cnnso,cnn_prob


# # box1 = gr.Textbox(label="Baseline model prediction")
# # box2 = gr.Textbox(label="Baseline prediction probability")

# box3 = gr.Textbox(label="CNN model prediction")
# box4 = gr.Textbox(label="CNN prediction probability")


# demo = gr.Interface(fn=check, inputs="text", outputs=[box3,box4])
# demo.launch(debug=True)
def gdp_change(text, model):
    a = ""
    b = ""
    prob1 = 0
    prob2 = 0
    if model == "CNN":
        cnn_model = CNN_model.CNNTextClassifier(glove,args)
        cnn_model.load_state_dict(torch.load('/Users/lifeifan/Desktop/ece1786/a2/A2_Gradio/cnn.pt'))

        tokens = text.split()
        token_ints = [glove.stoi.get(tok, len(glove.stoi)-1) for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1,1)
        cnn_pre = cnn_model(token_tensor)
        cnn_prob = float(torch.sigmoid(cnn_pre))
        if cnn_prob > 0.5:
            a,prob1 = "progressive", cnn_prob
        else:
            a,prob1 = "conservative", -(1-cnn_prob)
    if model == "BERT":
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bertmodel = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
        bertmodel.load_weights('/Users/lifeifan/Desktop/ece1786/a2/A2_Gradio/bert.h5')

        # Preprocess your single sentence
        # input_text = "Respect for women's rights and interests"
        input_text = text

        input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="tf")
        predictions = bertmodel(input_ids)


        logits = predictions.logits
        probabilities = tf.nn.softmax(logits, axis=1)
        prob1 = tf.reduce_max(probabilities)
        prob1 =  float(prob1.numpy())
        label = tf.argmax(probabilities[0])
        if label.numpy() == 0:
            a = "conservative"
            prob1 = -prob1
        else:
            a = "progressive"

    if model == "GPT2":
        gptmodel = GPT2ForSequenceClassification.from_pretrained("/Users/lifeifan/Desktop/ece1786/a2/A2_Gradio/gpt2")

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        inputs = tokenizer(text, return_tensors='pt')

        outputs = gptmodel(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

        softmax_logits = F.softmax(logits[0], dim=0)
        max_value, _ = torch.max(softmax_logits, dim=0)
        label_mapping = {0: 'conservative', 1: 'progressive'}
        predicted_class = label_mapping[predicted_labels.item()]
        a = predicted_class
        if a == "conservative":
            prob1 = -max_value.item()
        if a == "progressive":
            prob1 = max_value.item()
            


    fig, ax = plt.subplots()

    # Set the style
    sns.set_style("whitegrid")

    # Scatter plot
    ax.scatter(prob1, prob2, marker='o', s=400, edgecolors="red", facecolors="green")

    # Set title and labels
    ax.set_title("Political Leaning by " + model)
    ax.set_xlabel('Liberalism')
    ax.set_ylabel('Conservative', color='b')

    ax2 = ax.twiny()

    
    ax2.set_xlabel('Regulationism', color='r')
    # ax2.tick_params('x', colors='r')


    ax3 = ax.twinx()


    ax3.set_ylabel('Progressive', color='g')
    # ax3.tick_params('y', colors='g')

    # Add horizontal and vertical lines
    ax.axhline(0, color='black', linewidth=2, linestyle='-')
    ax.axvline(0, color='black', linewidth=2, linestyle='-')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    # Add grid
    # ax.grid()
    return a,prob1,prob2,fig

inputs = [
        "text",
        gr.Dropdown(['CNN', 'BERT', 'GPT2'], label="model")
    ]

box1 = gr.Textbox(label="CNN model prediction")
box2 = gr.Textbox(label="CNN prediction probability(social)")
box3 = gr.Textbox(label="CNN prediction probability(economic)")

figs = gr.Plot()

demo = gr.Interface(fn=gdp_change, inputs=inputs, outputs=[box1,box2,box3,figs])

demo.launch(debug=True)