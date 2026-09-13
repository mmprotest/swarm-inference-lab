"""Independent artifact/accounting audit; does not execute models or old suites."""
from pathlib import Path
import collections
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/E029_LOCAL_COMMIT_ANCHORED_WAN_TREE'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-5), (a, b)


def main():
    cfg = read('config.json')
    summary = read('summary.json')
    baseline = read('e028_baseline_import.json')
    correctness = read('correctness_results.json')
    prompt_ids = [p.stem for p in (OUT / 'references').glob('*.json')]
    assert len(prompt_ids) == len(set(prompt_ids)) == 8
    prompts = read('prompts.json')
    assert collections.Counter(x['category'] for x in prompts) == {'conversational': 2, 'coding': 2, 'reasoning': 2, 'long-context': 2}
    log = (OUT / 'logs/workload-2.log').read_text(encoding='utf-8', errors='replace')
    eog = {int(x) for x in re.findall(r'EOG token\s*=\s*(\d+)', log)}
    assert eog
    assert all(not (set(read(f'references/{p}.json')['committed_tokens']) & eog) for p in prompt_ids)
    expected = {
        f'{policy}-n{n}-d{d}-{net}-{prompt}.json'
        for n, d in ((8, 4), (16, 6), (32, 8))
        for policy in cfg['policies']
        for net in ([x['name'] for x in cfg['network_profiles']]
                    if policy == 'WAN_AWARE_TREE' else ['LOCAL'])
        for prompt in prompt_ids
    }
    files = list((OUT / 'runs').glob('*.json'))
    assert {p.name for p in files} == expected
    condensed = {}
    total_nodes = total_rounds = 0
    max_depth = max_nodes = 0
    for file in files:
        run = json.loads(file.read_text(encoding='utf-8'))
        ref = read(f'references/{run["prompt_id"]}.json')
        assert run['committed_tokens'] == ref['committed_tokens']
        assert len(run['committed_tokens']) == 256
        position = len(ref['prompt_tokens'])
        observed = []
        target = discarded = payload = local_ms = 0.0
        calls = 0
        for index, row in enumerate(run['rounds']):
            assert row['round_id'] == index
            assert row['root_position'] == position
            assert row['immutable_root'] and row['max_uncommitted_rounds'] == 1
            nodes = row['nodes']
            accepted = row['accepted_path']
            assert accepted and len(set(accepted)) == len(accepted)
            assert len(nodes) <= run['configuration']['node_budget']
            assert len({(n['parent_id'],n['token_id']) for n in nodes}) == len(nodes)
            assert [n['node_id'] for n in nodes] == list(range(len(nodes)))
            for node in nodes:
                parent = node['parent_id']
                assert -1 <= parent < node['node_id']
                assert node['depth'] == (1 if parent == -1 else nodes[parent]['depth'] + 1)
                assert node['depth'] <= run['configuration']['max_depth']
                assert node['verified']
                assert node['accepted'] == (node['node_id'] in accepted)
                assert node['rejected'] != node['accepted']
                assert node['status'] == ('accepted' if node['accepted'] else 'rejected')
            assert [nodes[i]['parent_id'] for i in accepted] == [-1] + accepted[:-1]
            tokens = [nodes[i]['token_id'] for i in accepted]
            assert tokens == row['committed_tokens']
            assert not any(n['parent_id'] == accepted[-1] and n['token_id'] == row['greedy'][accepted[-1]] for n in nodes)
            for a, b in zip(accepted, accepted[1:]):
                assert row['greedy'][a] == nodes[b]['token_id']
            observed.extend(tokens)
            position += len(tokens)
            stage_compute = sum(x['compute_ms'] for x in row['stages'] + row['commit'])
            assert stage_compute >= 0
            target += stage_compute
            off_path = 0.0
            for stage in row['stages']:
                assert stage['physical_decode_calls'] == len(nodes)
                assert sorted(i for x in stage['levels'] for i in x['nodes']) == list(range(len(nodes)))
                for level in stage['levels']:
                    off_path += (level['compute_ns'] + level['fork_ns']) / 1e6 * (
                        sum(i not in accepted for i in level['nodes']) / len(level['nodes']))
            close(off_path, row['discarded_compute_ms'])
            assert 0 <= off_path <= stage_compute
            discarded += off_path
            calls += sum(x['physical_decode_calls'] for x in row['stages'] + row['commit'])
            payload += sum(x['request_bytes'] for x in row['stages']) + row['stages'][-1]['response_bytes']
            payload += sum(x['request_bytes'] + x['response_bytes'] for x in row['commit'])
            # Independent closed-form zero-WAN schedule: serial traversal,
            # parallel commits on three independent stages, then next round.
            local_ms += row['cpu_overhead_ms'] + sum(x['service_ms'] for x in row['stages'])
            local_ms += max(x['service_ms'] for x in row['commit'])
            if run['configuration']['policy'] != 'ORACLE_TREE_UPPER_BOUND':
                local_ms += (row.get('draft') or {}).get('service_ms', 0)
                local_ms += (row.get('inject') or {}).get('service_ms', 0)
            total_nodes += len(nodes)
            max_nodes = max(max_nodes, len(nodes))
            max_depth = max(max_depth, max(x['depth'] for x in nodes))
        assert observed == run['committed_tokens']
        total_rounds += len(run['rounds'])
        condensed[str(file.relative_to(OUT))] = dict(target=target, discarded=discarded,
            payload=payload, local_ms=local_ms, calls=calls, rounds=len(run['rounds']))
    assert max_nodes == 32 and max_depth == 8
    groups = collections.defaultdict(list)
    sweep_count = 0
    with (OUT / 'wan_sweep_results.jsonl').open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            p = condensed[row['physical_source']]
            close(row['target_compute_ms'], p['target'])
            close(row['discarded_target_compute_ms'], p['discarded'])
            close(row['bytes_transferred'], p['payload'])
            assert row['physical_local_target_calls'] == p['calls']
            assert row['rounds'] == p['rounds']
            if row['network'] == 'LOCAL':
                close(row['elapsed_ms'], p['local_ms'])
            close(row['committed_tokens_per_second'], 256000 / row['elapsed_ms'])
            groups[(row['policy'], row['node_budget'], row['max_depth'], row['network'])].append(row)
            sweep_count += 1
    assert sweep_count == 1800 and len(groups) == 75
    for aggregate in read('wan_aggregate.json'):
        key = tuple(aggregate[k] for k in ('policy', 'node_budget', 'max_depth', 'network'))
        rows = groups[key]
        assert len(rows) == 24 and len({r['prompt_id'] for r in rows}) == 8
        assert len({(r['prompt_id'], r['seed']) for r in rows}) == 24
        elapsed = sum(r['elapsed_ms'] for r in rows)
        close(aggregate['committed_tokens_per_second'], 24 * 256000 / elapsed)
        close(aggregate['committed_tokens_per_wan_traversal'], 24 * 256 / sum(r['rounds'] for r in rows))
        close(aggregate['discarded_target_compute_fraction'], sum(r['discarded_target_compute_ms'] for r in rows) / sum(r['target_compute_ms'] for r in rows))
    branches = read('branch_state_correctness.json')
    assert branches['branch_state_tests'] == 16
    assert branches['branch_state_failures'] == branches['cross_branch_contamination_events'] == 0
    assert {(x['prompt_id'], x['siblings']) for x in branches['tests']} == {(p, n) for p in prompt_ids for n in (2, 4)}
    for test in branches['tests']:
        ref = read(f"references/{test['prompt_id']}.json")
        assert test['root_before'] == test['root_after']
        assert test['branch_commit'] == test['alternate_sibling_commit']
        assert test['continuation'] == ref['committed_tokens'][4:20]
        assert all(x['position'] == len(ref['prompt_tokens']) + 4 for x in test['branch_commit'])
        assert all(test['checks'].values())
    assert correctness['physical_runs'] == 216 and correctness['committed_tokens'] == 55296
    assert summary['physical_runs'] == 216 and summary['physical_committed_tokens'] == 55296
    for rel, digest in baseline['sha256'].items():
        assert hashlib.sha256((ROOT / baseline['source'] / rel).read_bytes()).hexdigest() == digest
    assert (OUT / 'prompts.json').read_bytes() == (ROOT / baseline['source'] / 'prompts.json').read_bytes()
    manifest_path = OUT / 'provenance/execution_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert hashlib.sha256((ROOT / '.runtime/experiment-029/build/llama-e029-stage.exe').read_bytes()).hexdigest() == manifest['native_binary_sha256']
        for relative, record in manifest['files'].items():
            frozen = OUT / 'provenance/executed_sources' / relative
            assert hashlib.sha256(frozen.read_bytes()).hexdigest() == record['sha256']
    original_config = OUT / 'provenance/preregistered_config.json'
    if original_config.exists():
        original = json.loads(original_config.read_text(encoding='utf-8'))
        assert original['preregistered_criteria'] == cfg['preregistered_criteria']
    assert len(list((OUT / 'plots').glob('*.png'))) == 7
    result = dict(passed=True, physical_runs=216, physical_committed_tokens=55296,
        physical_rounds=total_rounds, verified_nodes=total_nodes, wan_replays=sweep_count,
        aggregate_points=len(groups), maximum_tested_nodes=max_nodes, maximum_tested_depth=max_depth,
        zero_wan_closed_form_matches=True, rejected_work_recomputed=True,
        serialized_payload_totals_match=True, sealed_e028_unchanged=True,
        prompts_identical_to_e028=True, no_early_end_of_generation_tokens=True, no_model_execution_in_this_audit=True)
    (OUT / 'artifact_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
