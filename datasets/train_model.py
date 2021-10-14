# Main loop
import timm
import pytorch

model = timm.create_model('resnet18', pretrained=True)

