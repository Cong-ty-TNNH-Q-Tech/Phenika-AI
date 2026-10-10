"""Portable runner around the existing trained CV stack.

Run validation first; then choose policies per robot on the cached predicted
graphs. Test inference needs that selection file and never fits anything.
All outputs go in --out; original checkpoints/submissions are preserved.
"""
import argparse
import ast
import copy
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'solution/oracle'))
sys.path.insert(0, str(ROOT / 'solution/nlp'))
import policy_v5
from references import apply_mission
from policy_ranker import Policy


def legal_prediction(scene, prediction, robot):
    from oracle_xgb import adjacency, ACTIONS
    start = tuple(scene['robot']['rc'])
    legal = [ACTIONS[(v[0] - start[0], v[1] - start[1])]
             for v, _ in adjacency(scene, robot == 4)[start]]
    return int(prediction) if prediction in legal else (legal[0] if legal else 2)


class Capture:
    def __init__(self, model, selection, style_model=None):
        self.model, self.selection = model, selection
        self.style_model = style_model
        self.graphs = []

    def predict_all(self, graph):
        baseline = policy_v5.predict_all(graph)
        learned = self.model.predict_all(graph) if self.model else baseline
        self.last_learned = learned
        saved = copy.deepcopy(graph)
        saved['_scene_index'] = self.scene_index
        saved['_baseline'] = [legal_prediction(graph, baseline[r], r) for r in range(10)]
        saved['_ranker'] = [legal_prediction(graph, learned[r], r) for r in range(10)]
        self.graphs.append(saved)
        return {robot: legal_prediction(graph, learned[robot] if self.selection[robot] else baseline[robot], robot)
                for robot in range(10)}

    def greedy9(self, graph):
        if self.model and self.selection[9]:
            return self.predict_cached_robot9(graph)
        return policy_v5.greedy9(graph)

    def predict_cached_robot9(self, graph):
        return legal_prediction(graph, self.last_learned[9], 9)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    project = args.project.resolve()
    data = args.data.resolve()
    cv = args.models.resolve()
    names = ['nodes_frcnn.pt', 'legend_frcnn.pt', 'landmark_resnet18.pt',
             'robot_mobilenet.pt', 'weather_img.pt', 'stairs_mobilenet.pt',
             'oneway_dir.pt', 'edge_resnet18.pt', 'edge_presence.pt', 'siamese_mlp.pt']
    if args.joint_detector:
        names=names[2:]
    missing = [str(cv / name) for name in names if not (cv / name).exists()]
    if missing:
        raise FileNotFoundError('Missing CV checkpoints: ' + ', '.join(missing))
    observations = json.loads((data / args.split / 'observations.json').read_text(encoding='utf8'))
    # Read only for inference; test content is not logged or used to select models.
    missions = json.loads((project / f'solution/nlp/v3_{args.split}_missions.json').read_text(encoding='utf8'))
    if len(observations) != len(missions) * 10:
        raise ValueError('Mission/observation length mismatch')
    for i in range(len(missions)):
        rows = observations[i * 10:(i + 1) * 10]
        if [r['robot_id'] for r in rows] != list(range(10)):
            raise ValueError('Unexpected robot order')
    model = Policy(args.policy) if args.policy else None
    selection = [False] * 10
    if args.selection:
        selection = json.loads(args.selection.read_text())['use_ranker']
    style_model = None
    if args.style:
        import joblib
        style_model = joblib.load(args.style)
    capture = Capture(model, selection, style_model)
    if args.anchor_aliases:
        from anchor_miner import install
        install(args.anchor_aliases)
    source = (project / f'solution/pipeline/infer_{"val" if args.split == "validation" else "test"}_v5.py').read_text(encoding='utf8')
    source = source.replace('/mnt/hdd2/qtech/Phenika-AI/v3data/delivery_public', data.as_posix())
    source = source.replace('/mnt/hdd2/qtech/Phenika-AI/solution/cv', cv.as_posix())
    source = source.replace('/mnt/hdd2/qtech/Phenika-AI', project.as_posix())
    source = source.replace('from anchor_rule_v2 import predict_anchor as _predict_anchor', '_predict_anchor = _match_anchor')
    source = source.replace('from multileg import best_first_multileg', '# Removed unused external dependency')
    source = source.replace('import policy_v5 as _P5', '_P5 = capture')
    source = source.replace("_rk=_rule_kind(te_obs0[_i*10]['mission'])",
                            "_rk=_rule_kind(normalize_mission(te_obs0[_i*10]['mission']))")
    source = source.replace("_anc=_match_anchor(mission.get('mission_text',''),mission['goal'],mission['via'],_det_lms)",
                            "_anc=anchor_predict(mission.get('mission_text',''),mission,_det_lms)")
    if args.joint_detector:
        begin=source.index('nd=fasterrcnn_resnet50_fpn(')
        end=source.index('lm=models.resnet18(',begin)
        source=source[:begin]+"shared_detector=Views(joint_checkpoint,dev)\nnd=View(shared_detector,'nodes')\nlg=View(shared_detector,'legend')\n"+source[end:]
    source = source.replace("_m['mission_text']=te_obs0[_i*10]['mission']",
                            "_m.update(apply_flags(te_obs0[_i*10]['mission'], _m))\n    _m['mission_text']=te_obs0[_i*10]['mission']")
    # Preserve full graph and explicit goal/via references, including near/far.
    beginning = source.index('        # v3 ref resolution')
    end = source.index('        _P5preds =', beginning)
    replacement = '''        goal_type = mission['goal']; via_type = mission['via']
        if capture.style_model is not None:
            _style = capture.style_model.predict([style_features(im)])[0]
        fake={'nodes':[{'rc':list(rc),'xy':list(xy)} for rc,xy in rc2xy.items()],
              'edges':edge_list,'landmarks':[{'type':t,'rc':list(rc)} for rc,t in rc2lm.items()],
              'robot':{'rc':list(robot_rc),'heading':robot_hd},'weather':wthr,'style':_style}
        fake = apply_mission(fake, mission)
        goal_type = fake['mission']['goal']
'''
    source = source[:beginning] + replacement + source[end:]
    # Capture at the correct scene index including scenes that fail detection.
    source = source.replace('img_path=BASE/', 'capture.scene_index = si\n        img_path=BASE/', 1)
    # Avoid loading ImageNet weights just to overwrite them from a local state dict.
    source = source.replace('siam=Siam2()', 'siam=Siam2(pretrained=False)')
    source = source.replace('fasterrcnn_resnet50_fpn(weights=None)',
                            'fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)')
    output_name = 'predictions_optimized_validation.json' if args.split == 'validation' else 'predictions_optimized_test.json'
    source = re.sub(r"pathlib\.Path\('[^']+/solution/predictions_v5_(val|test)\.json'\)",
                    f"pathlib.Path({str(args.out / output_name)!r})", source)
    from style_fast import features as style_features
    from mission_flags import apply as apply_flags
    from nlp_fast import normalize as normalize_mission
    from anchor_miner import predict as anchor_predict
    namespace = {'__name__': '__main__', 'capture': capture,
                 'apply_mission': apply_mission, 'style_features': style_features}
    namespace['apply_flags'] = apply_flags
    namespace.update(normalize_mission=normalize_mission,anchor_predict=anchor_predict)
    if args.joint_detector:
        from joint_detector import Views,View
        namespace.update(Views=Views,View=View,joint_checkpoint=args.joint_detector)
    # The existing stack only sees predicted graph/mission, never val scene truth.
    exec(compile(source, str(project / 'solution/pipeline/portable_runtime.py'), 'exec'), namespace)
    preds = namespace['preds']
    if len(preds) != len(observations) or any(type(v) is not int or v not in range(4) for v in preds):
        raise ValueError('Invalid submission shape or values')
    (args.out / f'graphs_{args.split}.json').write_text(json.dumps(capture.graphs))
    parameter_counts = {name: sum(p.numel() for p in namespace[name].parameters())
                        for name in ['nd','lg','lm','rb','wx','st','owd','edg','pres','siam']}
    if args.joint_detector:
        parameter_counts['joint_detector']=sum(p.numel() for p in namespace['shared_detector'].model.parameters())
    parameter_counts['cv_total'] = sum(parameter_counts.values())
    parameter_counts['cached_nlp_checkpoint_not_loaded_here'] = True
    (args.out / 'cv_parameter_counts.json').write_text(json.dumps(parameter_counts, indent=2))
    if args.split == 'validation' and model:
        labels = np.array(json.loads((data / 'validation/labels.json').read_text())).reshape(-1, 10)
        # Selection must be computed on all rows, including CV failures.
        # Captures retain source index to avoid aligning a failed scene to the next one.
        score = {'macro': float(np.mean(np.array(preds).reshape(-1, 10) == labels)),
                 'per_robot': (np.array(preds).reshape(-1, 10) == labels).mean(0).tolist()}
        (args.out / 'e2e_metrics.json').write_text(json.dumps(score, indent=2))
        print(json.dumps(score), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=pathlib.Path, default=ROOT)
    p.add_argument('--data', type=pathlib.Path, required=True)
    p.add_argument('--models', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--policy', type=pathlib.Path)
    p.add_argument('--selection', type=pathlib.Path)
    p.add_argument('--style', type=pathlib.Path)
    p.add_argument('--joint-detector', type=pathlib.Path)
    p.add_argument('--anchor-aliases', type=pathlib.Path)
    main(p.parse_args())
