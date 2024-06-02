def format_duration(duration):
    hour = duration // (60 * 60)
    min = (duration % (60 * 60)) // 60
    second = duration % 60
    return f"{hour}: {min}: {second}"

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)