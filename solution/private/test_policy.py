import unittest
import numpy as np
from policy_ranker import route_scores, predict_model
from references import apply_mission


def edge(a,b,stairs=False,oneway=None,status='normal'):
    return {'a':a,'b':b,'stairs':stairs,'oneway_to':oneway,'status':status}


class PolicyTests(unittest.TestCase):
    def test_heading_is_preserved_through_via(self):
        # Move right to via, then down: one turn, not two from initial heading.
        scene={'nodes':[{'rc':[0,0]},{'rc':[0,1]},{'rc':[1,1]}],
               'edges':[edge([0,0],[0,1]),edge([0,1],[1,1])],
               'robot':{'rc':[0,0],'heading':'RIGHT'},
               'landmarks':[{'type':'dorm','rc':[0,1]},{'type':'lab','rc':[1,1]}],
               'mission':{'goal':'lab','goal_ref':None,'via':'dorm','via_ref':None}}
        scores=route_scores(scene,False,(1,1,2,20,1))
        self.assertEqual(scores[3],4.)
        self.assertTrue(np.isinf(scores[0]))

    def test_closed_oneway_and_stairs(self):
        scene={'nodes':[{'rc':[0,0]},{'rc':[0,1]}],
               'edges':[edge([0,0],[0,1],stairs=True)],
               'robot':{'rc':[0,0],'heading':'RIGHT'},
               'landmarks':[{'type':'lab','rc':[0,1]}],
               'mission':{'goal':'lab','goal_ref':None,'via':None,'via_ref':None}}
        self.assertTrue(np.isinf(route_scores(scene,False,(1,1,0,0,1))).all())
        self.assertEqual(route_scores(scene,True,(1,1,0,0,1))[3],1.)
        scene['edges'][0]['oneway_to']=[0,0]
        self.assertTrue(np.isinf(route_scores(scene,True,(1,1,0,0,1))).all())
        scene['edges'][0]['oneway_to']=None
        scene['edges'][0]['status']='closed'
        self.assertTrue(np.isinf(route_scores(scene,True,(1,1,0,0,1))).all())

    def test_reference_does_not_delete_landmarks(self):
        graph={'landmarks':[{'type':'library','rc':[0,0]},
                            {'type':'library','rc':[4,0]}, {'type':'dorm','rc':[4,2]}]}
        mission={'goal':'library','via':'library','urgent':False,'fragile':False,
                 'gref_kind':'north','via_ref_kind':'near','via_ref_anchor':'dorm'}
        result=apply_mission(graph,mission)
        self.assertEqual(result['mission']['goal_ref']['rc'],[0,0])
        self.assertEqual(result['mission']['via_ref']['rc'],[4,0])
        self.assertEqual(len(result['landmarks']),3)
        self.assertNotIn('mission',graph)

    def test_illegal_action_cannot_win(self):
        class Model:
            def predict_proba(self,x):
                return np.array([[.01,.99],[.1,.9],[.9,.1],[.9,.1]])
        pred=predict_model(Model(),np.zeros((1,4,2)),np.array([[False,True,True,False]]))
        self.assertEqual(pred.tolist(),[1])


if __name__=='__main__':
    unittest.main()
