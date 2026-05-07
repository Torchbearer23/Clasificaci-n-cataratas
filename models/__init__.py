from .mobilenet_v2 import get_mobilenet_v2
from .efficientnet_v2s import get_efficientnet_v2s
from .shufflenet_v2 import get_shufflenet_v2

def model_factory(model_name, input_shape=(224, 224, 3), num_classes=3):
    factory = {
        "mobilenet_v2": get_mobilenet_v2,
        "efficientnet_v2s": get_efficientnet_v2s,
        "shufflenet_v2": get_shufflenet_v2
    }
    
    if model_name not in factory:
        raise ValueError(f"Modelo {model_name} no soportado.")
        
    return factory[model_name](input_shape, num_classes)