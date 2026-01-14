import helpers.fcn_model as fcn_model
import helpers.segformer_model as segformer_model

def get_model(num_classes, checkpoint, model_type):
    if model_type == 'FCN':
        model = fcn_model.get_model(num_classes, checkpoint)
    if model_type == 'SegFormer':
        model = segformer_model.get_model(num_classes, checkpoint)
    return model
