"""核对演练资料的完整性及关键的输入关系。"""
from hashlib import sha256
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent / 'records'


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / 'catalog.json').read_bytes())
        self.records = {}
        for name, expected in self.catalog.items():
            raw = (ROOT / (name + '.jsonl')).read_bytes()
            self.assertEqual(sha256(raw).hexdigest(), expected['sha256'])
            rows = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual(len(rows), expected['rows'])
            self.assertEqual(len({row['id'] for row in rows}), len(rows))
            self.records[name] = rows

    def test_business_inputs(self):
        d = self.records
        self.assertGreater(len(d['games']),100)
        self.assertTrue(any(x['available_week'] > x['week'] for x in d['games']))
        self.assertTrue(any(x['follow_until'] for x in d['rounds']))
        region = {x['id']:x['region'] for x in d['players']}
        self.assertTrue(any(len({region[p] for s in x['lineups'].values() for p in s.values()})==2 for x in d['games'] if x['week']<=8))
        self.assertTrue(all(len({region[p] for s in x['lineups'].values() for p in s.values()})==1 for x in d['games'] if x['week']>=9))
