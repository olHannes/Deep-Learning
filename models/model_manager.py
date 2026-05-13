
from models import model1
from models import model2
from models import model3
from models import model4
from models import model5
from models import model6
from models import model7
from models import model8
from models import model9
from models import model10

def get_model(model_choice=0, input_shape=(256,256,3), num_classes=10):
    
    if model_choice == 1:
        return model1.build_model(input_shape, num_classes)
    elif model_choice == 2:
        return model2.build_model(input_shape, num_classes)
    elif model_choice == 3:
        return model3.build_model(input_shape, num_classes)
    elif model_choice == 4:
        return model4.build_model(input_shape, num_classes)
    elif model_choice == 5:
        return model5.build_model(input_shape, num_classes)
    elif model_choice == 6:
        return model6.build_model(input_shape, num_classes)
    elif model_choice == 7:
        return model7.build_model(input_shape, num_classes)
    elif model_choice == 8:
        return model8.build_model(input_shape, num_classes)
    elif model_choice == 9:
        return model9.build_model(input_shape, num_classes)
    elif model_choice == 10:
        return model10.build_model(input_shape, num_classes)
    else:
        raise ValueError("Invalid model-choice: " , model_choice)