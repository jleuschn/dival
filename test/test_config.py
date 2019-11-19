# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, mock_open, Mock, MagicMock
import json
from copy import deepcopy
from io import StringIO
from dival.config import CONFIG_FILENAME, get_config, set_config, CONFIG


class TestConfig(unittest.TestCase):
    def test_get_config(self):
        c = {
            'lodopab_dataset': {
                'data_path': '/path/to/lodopab'
            }
        }
        config_str = json.dumps(c)
        with patch('dival.config.open', mock_open(read_data=config_str)) as m:
            self.assertDictEqual(get_config(), c)
            m.assert_called_once_with(CONFIG_FILENAME, 'r')
            self.assertEqual(get_config('lodopab_dataset'),
                             c['lodopab_dataset'])
            m.assert_called_with(CONFIG_FILENAME, 'r')
            self.assertEqual(get_config('lodopab_dataset/'),
                             c['lodopab_dataset'])
            self.assertEqual(get_config('lodopab_dataset/data_path'),
                             c['lodopab_dataset']['data_path'])

    def test_set_config(self):
        c = {
            'a': {
                'b': '1',
                'c': {
                    'd': '2',
                    'e': '3'
                },
            },
            'f': '2'
        }
        config_str = json.dumps(c)

        class ExtStringIO(StringIO):
            def __init__(self, ext, *args, **kwargs):
                self.ext = ext
                super().__init__(*args, **kwargs)
                self.ext['str'] = self.getvalue()

            def close(self):
                self.ext['str'] = self.getvalue()
                super().close()
        ext = {'str': config_str}
        with patch('dival.config.open',
                   lambda *a, **kw: ExtStringIO(ext, ext['str'])):
            with patch('dival.config.CONFIG', deepcopy(c)) as mc:
                a_b_new = '4'
                set_config('a/b', a_b_new)
                c_updated = deepcopy(c)
                c_updated['a']['b'] = a_b_new
                self.assertDictEqual(mc, c_updated)
                self.assertDictEqual(json.loads(ext['str']), c_updated)
        ext = {'str': config_str}
        with patch('dival.config.open',
                   lambda *a, **kw: ExtStringIO(ext, ext['str'])):
            with patch('dival.config.CONFIG', deepcopy(c)) as mc:
                a_c_new = {'d': '5', 'e': '6'}
                set_config('a/c', a_c_new)
                c_updated = deepcopy(c)
                c_updated['a']['c'] = a_c_new
                self.assertDictEqual(mc, c_updated)
                self.assertDictEqual(json.loads(ext['str']), c_updated)
        ext = {'str': config_str}
        with patch('dival.config.open',
                   lambda *a, **kw: ExtStringIO(ext, ext['str'])):
            with patch('dival.config.CONFIG', deepcopy(c)) as mc:
                g_new = '7'
                set_config('g', g_new)
                c_updated = deepcopy(c)
                c_updated['g'] = g_new
                self.assertDictEqual(mc, c_updated)
                self.assertDictEqual(json.loads(ext['str']), c_updated)


if __name__ == '__main__':
    unittest.main()
