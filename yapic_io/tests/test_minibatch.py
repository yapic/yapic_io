from unittest import TestCase
import yapic_io.minibatch as mb
from numpy.testing import assert_array_equal, assert_array_almost_equal
import os

base_path = os.path.dirname(__file__)


class TestMinibatch(TestCase):

    def test_tuple_checks(self):

        assert mb._is_twotuple_of_numerics((2, 3))
        assert mb._is_twotuple_of_numerics((2., 3.))
        assert not mb._is_twotuple_of_numerics((2., 3., 4))
        assert not mb._is_twotuple_of_numerics(('text', 3))
        assert not mb._is_twotuple_of_numerics(4)

        assert mb._is_list_of_twotuples([(2, 3), (4, 6)])
        assert mb._is_list_of_twotuples([(2, 3)])
        assert not mb._is_list_of_twotuples([((2, 3), 3), (4, 2)])
        assert not mb._is_list_of_twotuples([(2, 3), 'text'])
