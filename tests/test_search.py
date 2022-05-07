from pytest import mark
from backdoor import search
from helpers import rand

@rand(10)
def test_uniform():
    p = search.Uniform(1, 10)
    assert p.sample() < 10
    assert p.sample() >= 1

@rand(10)
def test_uniform_int():
    p = search.Uniform(1, 10, integer=True)
    assert (x := p.sample()) == int(x)

@rand(10)
def test_loguniform():
    p = search.LogUniform(1, 10)
    assert p.sample() < 10
    assert p.sample() >= 1

@rand(10)
def test_loguniform_int():
    p = search.LogUniform(1, 10, integer=True)
    assert (x := p.sample()) == int(x)

@rand(10)
def test_boolean():
    p = search.Boolean()
    assert p.sample() in [True, False]

@rand(10)
def test_choice():
    p = search.Choice([1, 2, 3])
    assert p.sample() in [1, 2, 3]

def test_bsonify():
    class TestClass:
        def __repr__(self):
            return 'TestClass'

    d = {"1": 1, "2": TestClass()}
    bson = search.bsonify(d)
    assert bson['1'] == 1
    assert bson['2'] == 'TestClass'

def test_searchable_passthrough():
    s = search.Searchable(lambda x: x)
    assert s(1) == 1

def test_random_search():
    s = search.Searchable(lambda x: x)

    s.random_search([search.Choice([1, 2, 3])], {}, trials=3)