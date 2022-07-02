from torch.nn import Module


def count_parameters(model: Module) -> int:
    return sum(p.numel() for p in model.parameters())
