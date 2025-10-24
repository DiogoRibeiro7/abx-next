from abx_next import srm_test

def test_srm_balanced():
    res = srm_test(1000, 1000)
    assert res["pvalue"] > 0.05
