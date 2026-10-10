"""Resolve goal/via references without deleting other landmark instances."""
import copy
import math


def resolve(landmarks, kind, target_type, anchor):
    all_points = [(lm['type'], tuple(lm['rc'])) for lm in landmarks]
    points = [point for typ, point in all_points if typ == target_type]
    if kind in ['north_most', 'south_most', 'west_most', 'east_most']:
        pool = all_points
        direction = kind.replace('_most', '')
    else:
        pool = [(target_type, point) for point in points]
        direction = kind
    axis = {'north': (0, 1), 'south': (0, -1), 'west': (1, 1), 'east': (1, -1)}
    if direction in axis and pool:
        dim, sign = axis[direction]
        typ, point = min(pool, key=lambda pair: (pair[1][dim] * sign, pair[1][1 - dim]))
        return typ, point
    if kind in ['near', 'far', 'anchor_near']:
        anchors = [point for typ, point in all_points if typ == anchor]
        if len(anchors) != 1:
            return target_type, None
        if kind == 'anchor_near':
            pool = [(typ, point) for typ, point in all_points if typ != anchor]
        if pool:
            origin = anchors[0]
            sign = -1 if kind == 'far' else 1
            typ, point = min(pool, key=lambda pair:
                             (sign * math.hypot(pair[1][0] - origin[0], pair[1][1] - origin[1]), pair[1]))
            return typ, point
    return target_type, None


def apply_mission(graph, parsed):
    """Convert NLP outputs to policy schema; preserve unrelated duplicate landmarks."""
    scene = copy.deepcopy(graph)
    mission = {'goal': parsed['goal'], 'via': parsed['via'],
               'urgent': bool(parsed['urgent']), 'fragile': bool(parsed['fragile'])}
    for key, kind_key, anchor_key in [('goal', 'gref_kind', 'gref_anchor'),
                                     ('via', 'via_ref_kind', 'via_ref_anchor')]:
        kind = parsed.get(kind_key, 'NONE')
        anchor = parsed.get(anchor_key)
        typ, point = resolve(scene['landmarks'], kind, parsed[key], anchor)
        mission[key] = typ
        mission[key + '_ref'] = ({'kind': kind, 'anchor': anchor, 'rc': list(point)}
                                  if point is not None else None)
    scene['mission'] = mission
    return scene
