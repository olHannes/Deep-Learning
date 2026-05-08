
from models import model1
from models import model2

def get_model(model_choice=0, input_shape=(256,256,3), num_clases=10):
    
    if model_choice == 1:
        return model1.build_model_1(input_shape, num_clases)
    elif model_choice == 2:
        return model2.build_model_2(input_shape, num_clases)
    else:
        raise ValueError("Invalid model-choice: " , model_choice)