"""Cross-validate Layer 1 facts against GPT-5.2 independent analysis.

Feeds raw XYZ files (not our processed data) to GPT-5.2 and compares
the LLM's independent analysis against our pipeline's JSON output.
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

from openai import OpenAI


SYSTEM = """You are a computational chemist expert. You are given a raw XYZ file of a molecular building block used in MOF (Metal-Organic Framework) assembly software called PORMAKE.

XYZ FILE FORMAT:
- Line 1: number of atoms
- Line 2: comment/header (may be blank)
- Lines 3+: atom data, one per line
  - Columns: Element X Y Z
  - Element "X" = dummy atom (connection point where this fragment connects to other fragments in the MOF)
- After the atom coordinates, there is a BOND BLOCK section:
  - Each line: atom_index1 atom_index2 bond_type
  - atom indices are 1-based
  - bond_type: S=single, D=double, T=triple, A=aromatic (Note: PORMAKE uses "A" for resonance-delocalized bonds like carboxylate C-O, nitro N-O, not just true aromatic rings)

TASK: Analyze the XYZ file independently and report:
1. formula: Molecular formula in Hill notation (exclude dummy atoms X)
2. molecular_weight: in Daltons (exclude X atoms)
3. total_atoms: count of real atoms (exclude X)
4. connection_points: how many X (dummy) atoms
5. metals: list of metal elements present (empty list if none)
6. num_rings: count of rings in the molecule (use minimum cycle basis count)
7. smiles: SMILES string with [*] for connection points
8. functional_groups: list of all functional groups/chemical features you can identify
9. readable_name: a concise human-friendly name for this molecule (2-8 words)
10. is_rigid: true if no freely rotatable single bonds exist between connection points (biaryl single bonds count as rotatable)

Respond in JSON only:
{"formula": "...", "molecular_weight": 0.0, "total_atoms": 0, "connection_points": 0, "metals": [], "num_rings": 0, "smiles": "...", "functional_groups": [], "readable_name": "...", "is_rigid": false}"""


TEST_BBS = ['E1', 'E101', 'E118', 'E52', 'E22', 'E129',
            'N1', 'N100', 'N102', 'N10', 'N428', 'N200']


def run_crossval():
    sys.stdout.reconfigure(encoding='utf-8')
    client = OpenAI()

    bbs_dir = Path('_source_data/bbs')
    v8_dir = Path('bb_metadata_v8')
    results = []

    for bb_id in TEST_BBS:
        xyz_path = bbs_dir / f'{bb_id}.xyz'
        json_path = v8_dir / f'{bb_id}.json'

        xyz_content = xyz_path.read_text(encoding='utf-8')
        our_data = json.loads(json_path.read_text(encoding='utf-8'))

        print(f'Processing {bb_id}...', flush=True)

        try:
            response = client.chat.completions.create(
                model='gpt-5.2',
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Analyze this XYZ file:\n\n{xyz_content}"},
                ],
                temperature=0.1,
                max_completion_tokens=1000,
                response_format={"type": "json_object"},
            )
            llm_result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

        l1 = our_data['layer1_facts']
        l2 = our_data['layer2_semantics']

        discrepancies = []

        if llm_result.get('formula', '') != l1['formula']:
            discrepancies.append(
                f"FORMULA: ours={l1['formula']}, LLM={llm_result.get('formula')}")

        llm_mw = llm_result.get('molecular_weight', 0)
        if abs(llm_mw - l1['molecular_weight']) > 1.0:
            discrepancies.append(
                f"MW: ours={l1['molecular_weight']:.1f}, LLM={llm_mw:.1f}")

        if llm_result.get('total_atoms', 0) != l1['total_atoms']:
            discrepancies.append(
                f"ATOMS: ours={l1['total_atoms']}, LLM={llm_result.get('total_atoms')}")

        if llm_result.get('connection_points', 0) != l1['connection_points']['count']:
            discrepancies.append(
                f"CP: ours={l1['connection_points']['count']}, "
                f"LLM={llm_result.get('connection_points')}")

        our_metals = sorted(l1['metals'])
        llm_metals = sorted(llm_result.get('metals', []))
        if our_metals != llm_metals:
            discrepancies.append(
                f"METALS: ours={our_metals}, LLM={llm_metals}")

        if llm_result.get('num_rings', 0) != l1['num_rings']:
            discrepancies.append(
                f"RINGS: ours={l1['num_rings']}, LLM={llm_result.get('num_rings')}")

        if llm_result.get('is_rigid') != l1['is_rigid']:
            discrepancies.append(
                f"RIGID: ours={l1['is_rigid']}, LLM={llm_result.get('is_rigid')}")

        our_name = l2.get('readable_name', '')
        llm_name = llm_result.get('readable_name', '')

        result_entry = {
            'bb_id': bb_id,
            'discrepancies': discrepancies,
            'our_formula': l1['formula'],
            'llm_formula': llm_result.get('formula'),
            'our_mw': l1['molecular_weight'],
            'llm_mw': llm_result.get('molecular_weight'),
            'our_atoms': l1['total_atoms'],
            'llm_atoms': llm_result.get('total_atoms'),
            'our_cp': l1['connection_points']['count'],
            'llm_cp': llm_result.get('connection_points'),
            'our_metals': our_metals,
            'llm_metals': llm_metals,
            'our_rings': l1['num_rings'],
            'llm_rings': llm_result.get('num_rings'),
            'our_rigid': l1['is_rigid'],
            'llm_rigid': llm_result.get('is_rigid'),
            'our_name': our_name,
            'llm_name': llm_name,
            'our_smiles': l1['smiles'],
            'llm_smiles': llm_result.get('smiles'),
            'our_fgs': l2['functional_groups']['rule_based'],
            'llm_fgs': llm_result.get('functional_groups', []),
        }
        results.append(result_entry)

        status = "MATCH" if not discrepancies else f"MISMATCH ({len(discrepancies)})"
        print(f'  {status}', flush=True)
        if discrepancies:
            for d in discrepancies:
                print(f'    {d}')
        print(f'  Our name: {our_name}')
        print(f'  LLM name: {llm_name}')
        print(f'  Our FGs:  {l2["functional_groups"]["rule_based"]}')
        print(f'  LLM FGs:  {llm_result.get("functional_groups", [])}')
        print(flush=True)
        time.sleep(0.5)

    # Summary
    print('\n' + '=' * 70)
    print('CROSS-VALIDATION SUMMARY (GPT-5.2 vs Pipeline)')
    print('=' * 70)
    total = len(results)
    perfect = sum(1 for r in results if not r['discrepancies'])
    print(f'Total checked: {total}')
    print(f'Perfect match: {perfect}/{total}')
    print(f'With discrepancies: {total - perfect}/{total}')

    disc_types = Counter()
    for r in results:
        for d in r['discrepancies']:
            field = d.split(':')[0]
            disc_types[field] += 1

    if disc_types:
        print('\nDiscrepancy breakdown:')
        for field, count in disc_types.most_common():
            print(f'  {field}: {count}/{total} BBs')

    print('\n--- Per-BB Detail ---')
    for r in results:
        marker = 'OK' if not r['discrepancies'] else 'XX'
        print(f"[{marker}] {r['bb_id']}: {r['our_formula']}")
        if r['discrepancies']:
            for d in r['discrepancies']:
                print(f"     {d}")
        if r['our_name'] != r['llm_name']:
            print(f"     NAME: ours='{r['our_name']}', LLM='{r['llm_name']}'")
        our_set = set(r['our_fgs'])
        llm_set = set(r['llm_fgs'])
        only_ours = our_set - llm_set
        only_llm = llm_set - our_set
        if only_ours:
            print(f"     FGs only in ours: {sorted(only_ours)}")
        if only_llm:
            print(f"     FGs only in LLM:  {sorted(only_llm)}")

    report_path = Path('bb_metadata_v8/_crossval_report.json')
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nFull report: {report_path}')


if __name__ == '__main__':
    run_crossval()
