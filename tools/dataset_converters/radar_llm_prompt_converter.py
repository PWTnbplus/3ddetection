from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


SYSTEM_PROMPT = (
    'You are generating concise driving-scene text for a 3D object detection '
    'dataset. Use only the provided detector outputs. Do not invent objects, '
    'colors, road types, weather, traffic lights, or object attributes that '
    'are not present in the input. Keep the description useful for downstream '
    'multi-modal 3D detection.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build LLM prompts from radar detector text records.')
    parser.add_argument(
        '--input-file',
        default='/root/lanyun-fs/dataset/radar/texts/radar_texts_prediction.json',
        help='Input JSON generated from radar detector predictions.')
    parser.add_argument(
        '--output-file',
        default=None,
        help='Output JSONL prompt file. Defaults to '
        '<input-dir>/radar_llm_prompts_prediction.jsonl.')
    parser.add_argument(
        '--max-objects',
        type=int,
        default=30,
        help='Maximum number of high-confidence objects included in a prompt.')
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.0,
        help='Optional extra score threshold for objects in prompts.')
    parser.add_argument(
        '--language',
        choices=['en', 'zh'],
        default='en',
        help='Language of the expected LLM scene description.')
    return parser.parse_args()


def load_json(path: Path) -> List[Dict]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def dump_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def get_object_score(obj: Dict) -> float:
    score = obj.get('score')
    if score is None:
        return 1.0
    return float(score)


def get_object_location(obj: Dict) -> List[float]:
    if 'location_lidar' in obj:
        return obj['location_lidar']
    return obj.get('location_camera', [])


def format_location(obj: Dict) -> str:
    location = get_object_location(obj)
    if len(location) >= 3:
        coord_type = 'LiDAR' if 'location_lidar' in obj else 'camera'
        return (
            f'{coord_type}=({location[0]:.2f}, {location[1]:.2f}, '
            f'{location[2]:.2f})')
    return 'location=unknown'


def summarize_record(record: Dict) -> Dict:
    objects = record.get('objects', [])
    class_counter = Counter(obj.get('class_name', 'object') for obj in objects)
    nearest = None
    for obj in objects:
        if nearest is None or obj.get('distance', 1e9) < nearest.get(
                'distance', 1e9):
            nearest = obj

    return dict(
        sample_id=record.get('sample_id', ''),
        num_objects=len(objects),
        counts_by_class=dict(sorted(class_counter.items())),
        counts_by_direction=dict(
            sorted(record.get('counts_by_direction', {}).items())),
        nearest_object=nearest)


def build_object_lines(record: Dict, max_objects: int,
                       min_score: float) -> List[str]:
    objects = [
        obj for obj in record.get('objects', [])
        if get_object_score(obj) >= min_score
    ]
    objects = sorted(
        objects,
        key=lambda obj: (-get_object_score(obj), obj.get('distance', 1e9)))

    lines = []
    for obj in objects[:max_objects]:
        parts = [
            f'class={obj.get("class_name", "object")}',
            f'distance={obj.get("distance", 0.0):.1f}m',
            f'score={get_object_score(obj):.3f}',
            format_location(obj),
        ]
        lines.append('- ' + ', '.join(parts))
    return lines


def build_user_prompt(record: Dict, max_objects: int, min_score: float,
                      language: str) -> str:
    summary = summarize_record(record)
    object_lines = build_object_lines(record, max_objects, min_score)

    if language == 'zh':
        instruction = (
            '请根据下面的雷达/点云检测结果，生成一句简洁的中文自动驾驶场景描述。'
            '只描述输入中出现的检测目标、数量、距离和大致空间分布；不要编造颜色、天气、'
            '车道线、交通灯或不存在的目标。输出只保留一句话。')
    else:
        instruction = (
            'Generate one concise English driving-scene description from the '
            'radar/point-cloud detections below. Mention object types, counts, '
            'distance distribution, and coarse spatial layout when useful. Do '
            'not invent missing objects or visual attributes. Output exactly '
            'one sentence.')

    prompt_lines = [
        instruction,
        '',
        f'Sample id: {summary["sample_id"]}',
        f'Object counts: {json.dumps(summary["counts_by_class"], ensure_ascii=False)}',
        f'Direction counts: {json.dumps(summary["counts_by_direction"], ensure_ascii=False)}',
        f'Template baseline: {record.get("text", "")}',
        'Detections:',
    ]
    if object_lines:
        prompt_lines.extend(object_lines)
    else:
        prompt_lines.append('- no detected objects above the score threshold')
    return '\n'.join(prompt_lines)


def build_prompt_record(record: Dict, max_objects: int, min_score: float,
                        language: str) -> Dict:
    return dict(
        sample_id=record.get('sample_id', ''),
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(record, max_objects, min_score,
                                      language),
        template_text=record.get('text', ''),
        metadata=dict(
            source=record.get('source', ''),
            num_objects=record.get('num_objects', 0),
            counts_by_class=record.get('counts_by_class', {}),
            counts_by_direction=record.get('counts_by_direction', {})))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file) if args.output_file else (
        input_path.parent / 'radar_llm_prompts_prediction.jsonl')

    records = load_json(input_path)
    prompt_records = [
        build_prompt_record(record, args.max_objects, args.min_score,
                            args.language) for record in records
    ]
    dump_jsonl(prompt_records, output_path)
    print(f'Saved {len(prompt_records)} LLM prompts to {output_path}')


if __name__ == '__main__':
    main()
