def reserve(buffer, new_size):
    if new_size > len(buffer):
        buffer.extend([None] * (new_size - len(buffer)))
    return buffer
