def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    
    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps
    
    Returns:
        List of learning rates for each step
    """
    if num_steps > 1:
        step_size = (initial_lr-end_lr)/(num_steps-1)
        return [(initial_lr - t*step_size) for t in range(num_steps)]
    elif num_steps == 1:
        return [initial_lr]
    else:
        return []