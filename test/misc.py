import ddptensor as dt


def arange_reshape(a, b, s, shp, dtype=dt.float64):
    return dt.reshape(dt.arange(a, b, s, dtype), shp)
