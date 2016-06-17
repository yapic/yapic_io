import os

skip_long_running_tests = os.getenv('YAPIC_TEST_LONG_RUNNING', '0') == '0' or \
                          (len(os.getenv('YAPIC_TEST_LONG_RUNNING', '0')) == 0)

