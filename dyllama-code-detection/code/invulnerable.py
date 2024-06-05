def reserve(buffer, new_size):
    old_size = len(buffer)
    if new_size > old_size:
        buffer.extend([None] * (new_size - old_size))
    return buffer
