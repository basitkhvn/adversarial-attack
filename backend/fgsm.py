import torch
import torch.nn as nn

class Attack:
    def __init__(self,model,dev):
        self.model=model
        self.dev=dev

    def attack_function(self, epsilon, image, gradient):
        signed_gradient= torch.sign(gradient)
        changed_image= image + (epsilon * signed_gradient)
        changed_image= torch.clamp(changed_image,0,1)

        return changed_image