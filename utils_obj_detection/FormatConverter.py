def LTWH2YOLOv4Format(coord, W, H):
    return [(coord[0] + coord[2] / 2)/W, (coord[1] + coord[3] / 2) / H, coord[2] / W, coord[3] / H]