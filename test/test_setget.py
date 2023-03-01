import ddptensor as dt


class TestSetGet:
    def test_setitem1(self):
        a = dt.ones([16, 16], dt.float64)
        a[0:16:2, 0:16:2] = dt.zeros([8, 8], dt.float64)
        for i in range(16):
            for j in range(16):
                assert float(a[i : i + 1, j : j + 1]) == (1 if i % 2 or j % 2 else 0)

    def test_setitem2(self):
        a = dt.ones([16, 16], dt.float64)
        b = dt.zeros([16, 16], dt.float64)
        a[1:8, 0:6] = b[0:7, 0:6]
        for i in range(16):
            for j in range(16):
                if i >= 1 and i < 8 and j < 6:
                    assert float(a[i : i + 1, j : j + 1]) == 0
                else:
                    assert float(a[i : i + 1, j : j + 1]) == 1

    def test_setitem3(self):
        a = dt.ones([16, 16], dt.float64)
        b = dt.zeros([16, 16], dt.float64)
        a[7:16:3, 4:10:2] = b[4:7, 10:16:2]
        for i in range(16):
            for j in range(16):
                if (
                    i >= 7
                    and i < 16
                    and ((i - 7) % 3 == 0)
                    and j >= 4
                    and j < 10
                    and ((j - 4) % 2 == 0)
                ):
                    assert float(a[i : i + 1, j : j + 1]) == 0
                else:
                    assert float(a[i : i + 1, j : j + 1]) == 1
