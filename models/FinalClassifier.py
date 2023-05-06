from torch import nn

#this is the initialization line in the original code:
#models[m] = getattr(model_list, args.models[m].model)()
class Classifier(nn.Module):
    def __init__(self, dim_input, num_classes):
        super().__init__()
        self.classifier = nn.Linear(dim_input, num_classes)
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
#should return logits and features
#features is ignored for now
    def forward(self, x):
        return self.classifier(x), {}
