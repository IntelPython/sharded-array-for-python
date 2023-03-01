import ddptensor as dt


class TestManip:
    def test_reshape(self):
        a = dt.arange(0, 12 * 11, 1, dt.int64)
        c = dt.reshape(a, [12, 11])
        c = dt.reshape(a, [6, 22])
        dt.sync()
        for i in range(6):
            for j in range(22):
                assert int(c[i : i + 1, j : j + 1]) == i * 22 + j
