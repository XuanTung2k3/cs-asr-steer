"""Track B: learned interventions against a strong CS adapter baseline.

Every arm lives in one PyTorch / HuggingFace stack: the same model wrapper, the
same scorer, the same decode config, the same environment. No ESPnet, no second
environment, no cross-framework comparison -- a B3-vs-B1/B2 difference must be
attributable to the method, not to the framework it ran in.
"""
