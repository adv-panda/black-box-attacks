import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt 

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models 

from PIL import Image
from torchvision import transforms

from art.estimators.classification import PyTorchClassifier 

from art.attacks.evasion import SimBA 
from art.attacks.evasion import HopSkipJump 
from art.attacks.evasion import BoundaryAttack  
from art.attacks.evasion import ProjectedGradientDescent 

## Loading selected model from PyTorch models 
@st.cache()
def load_model(model_name): 
	if model_name == "VGG19": 		
		model = models.vgg19(pretrained=True)
	elif model_name == "ResNet152": 
		model = models.resnet152(pretrained=True)
	elif model_name == "DenseNet161": 
		model = models.densenet161() 
	elif model_name == "InceptionV3": 
		model = models.inception_v3() 
	elif model_name == "Xception": 
		pass 
	elif model_name == "GoogLeNet": 
		model = models.googlenet()
	elif model_name == "MobileNetV2":
		model = models.mobilenet_v2() 

	return model 


def generate_adv_example(attack, classifier, image):   
	if attack == "SimBA": 
		pgd_attack = ProjectedGradientDescent(classifier, max_iter=20, eps_step=1, eps=0.01)  
		# attack_simba = SimBA(classifier=classifier, epsilon = 0.05, max_iter=5000) 
		return pgd_attack.generate(x=image)  
	elif attack == "HopSkipJump": 
		hopskipjump_attack = HopSkipJump(classifier=classifier, max_iter=20, verbose=False) 
		return hopskipjump_attack.generate(x=image)  
	elif attack == "Boundary Attack": 
		boundary_attack = BoundaryAttack(estimator = classifier, epsilon=0.01, max_iter=1000, targeted=False, verbose=False)  
		return boundary_attack.generate(x=image)    


def softmax_activation(inputs): 
    inputs = inputs.tolist()
    exp_values = np.exp(inputs - np.max(inputs)) 
    # Normalize 
    probabilities = exp_values / np.sum(exp_values)

    return probabilities

## Preprocessing the image 
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])  


st.title('Black-box attacks and defenses')  
st.markdown("""
This app performs testing for adversarial attacks and defenses  
* **🐍 Python libraries:** ART, pytorch, streamlit, numpy, matplotlib
""")

st.sidebar.header('Parameters')  

image = st.sidebar.file_uploader('Upload a photo')

model_selection = st.sidebar.selectbox('Choose model: ', ['VGG19', 'ResNet152', 'DenseNet161', 'InceptionV3', 'Xception', 'GoogLeNet', 'MobileNetV2']) 
attack_selection = st.sidebar.radio('Choose attack: ', ['SimBA', 'HopSkipJump', 'Boundary Attack']) 
defense_selection = st.sidebar.radio('Choose defense: ', ['Bits Squeezing', 'Median Smoothing', 'JPEG Filter']) 

if st.sidebar.button('Get Results'): 
    input_image = Image.open(image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).numpy().astype(np.float32)

    selected_model = load_model(model_selection) 
    criterion = nn.CrossEntropyLoss()

	## Creating ART classifier
    classifier = PyTorchClassifier(
	    model=selected_model,
	    loss=criterion,
	    input_shape=(3, 224, 224),
	    nb_classes=1000,
	    device_type='gpu'
	)

    st.subheader("Benign Prediction")  
    st.write("Selected Model: " + model_selection) 

	## Making benign prediction 
    preds = classifier.predict(input_batch)
    pred_label = np.argmax(preds, axis=1)[0]
    st.write("Predicted label: " + str(pred_label))

    accuracy = np.max(softmax_activation(preds), axis=1)
    accuracy = round(accuracy[0], 2)
    accuracy_text = "Accuracy on benign examples: {}%".format(accuracy * 100)

    st.write(accuracy_text)

    st.subheader("Adversarial Attack") 
    st.write("Selected Attack: " + attack_selection)
    adv_image = generate_adv_example(attack_selection, classifier, input_batch)

	## Making adversarial prediction 
    adv_preds = classifier.predict(adv_image)
    adv_pred_label = np.argmax(adv_preds, axis=1)[0]
    st.write("Adversarial predicted label: " + str(adv_pred_label)) 
    adv_accuracy = np.max(softmax_activation(adv_preds), axis=1)[0]
    adv_accuracy_text = "Confidence on adversarial examples: {}%".format(round(adv_accuracy * 100, 2)) 
    st.write(adv_accuracy_text)

    if pred_label != adv_pred_label: 
    	st.write("Attack status: success") 
    else: 
    	st.write("Attack status: fail")  

    bedign_input = input_batch[0].transpose((1,2,0)) 
    adv_input = adv_image[0].transpose((1,2,0)) 

    figure, axarr = plt.subplots(1, 2, sharex=True, sharey=False) 
    axarr[0].imshow(bedign_input)
    axarr[1].imshow(adv_input)  

    st.pyplot(figure) 




